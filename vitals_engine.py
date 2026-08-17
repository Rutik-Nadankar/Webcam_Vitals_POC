"""Fast single-pass camera analysis with local proxy and result caching."""
from pathlib import Path
import hashlib,json,time,shutil,subprocess,csv
import cv2
import numpy as np
from scipy.signal import butter,filtfilt,detrend,welch,find_peaks
BASE_DIR=Path(__file__).resolve().parent; CASCADE_PATH=BASE_DIR/"haarcascade_frontalface_default.xml"; CACHE_DIR=BASE_DIR/"results"/"cache"
ANALYSIS_CACHE_VERSION = "v7"
TARGET_ANALYSIS_FPS = 30.0
RPPG_ANALYSIS_WIDTH = 800
RPPG_LARGE_FACE_WIDTH = 960
MIN_FACE_WIDTH_PIXELS = 90
MOTION_WIDTH = 480
FACE_DETECTION_INTERVAL = 10
MAX_ANALYSIS_SECONDS = 60.0
HR_LOW,HR_HIGH,RR_LOW,RR_HIGH=.75,2.5,.10,.50
def _notify(cb,stage,pct):
    if cb: cb(stage,pct)
def _empty(warnings,duration=0):
    return {"video_supported":False,"video_processing_status":"FAILED","heart_rate":None,"heart_rate_quality":"LOW","respiratory_rate":None,"respiratory_quality":"LOW","rmssd":None,"sdnn":None,"hrv_quality":"LOW","accepted_beats":0,"face_detection_percent":0.0,"measurement_duration":round(duration,1),"signal_quality_score":0,"signal_quality_label":"LOW","warnings":warnings,"timing":{}}
def _key(path):
    s=path.stat(); return hashlib.sha256(f"{ANALYSIS_CACHE_VERSION}|{path.resolve()}|{s.st_size}|{s.st_mtime_ns}".encode()).hexdigest()[:16]
def _open_capture(source):
    """Request hardware decoding when this OpenCV/FFmpeg build supports it; otherwise fall back safely."""
    available=hasattr(cv2,"CAP_PROP_HW_ACCELERATION") and hasattr(cv2,"VIDEO_ACCELERATION_ANY"); requested=False; used=False; backend="default"
    try:
        if available:
            requested=True
            cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_ANY])
            if cap.isOpened():
                backend="FFMPEG"; used=cap.get(cv2.CAP_PROP_HW_ACCELERATION)==cv2.VIDEO_ACCELERATION_ANY
                return cap,{"video_backend":backend,"hardware_decode_requested":requested,"hardware_decode_available":available,"hardware_decode_used":used}
            cap.release()
    except (cv2.error,TypeError):
        pass
    cap=cv2.VideoCapture(str(source))
    try: backend=cap.getBackendName()
    except cv2.error: pass
    return cap,{"video_backend":backend,"hardware_decode_requested":requested,"hardware_decode_available":available,"hardware_decode_used":used}
def create_optimized_demo_copy(video_path):
    """Optional manual FFmpeg helper; never called by normal analysis."""
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:return None
    source=Path(video_path).resolve();CACHE_DIR.mkdir(parents=True,exist_ok=True);output=CACHE_DIR/f"{source.stem}_720p30.mp4"
    if output.exists():return output
    subprocess.run([ffmpeg,"-y","-i",str(source),"-vf","scale=-2:720,fps=30","-c:v","libx264","-preset","ultrafast","-crf","23","-an",str(output)],check=True)
    return output
def _filter(signal,fs,low,high):
    if len(signal)<max(20,fs*5) or fs<=2*high:return None
    b,a=butter(4,[low/(fs/2),high/(fs/2)],btype="band")
    try:return filtfilt(b,a,detrend(signal))
    except ValueError:return None
def _pos(rgb):
    x=np.asarray(rgb,float); m=x.mean(0);m[m==0]=1;x=x/m;p,q=x[:,1]-x[:,2],x[:,1]+x[:,2]-2*x[:,0];return detrend(p+(np.std(p)/max(np.std(q),1e-8))*q)
def _roi(frame,face):
    x,y,w,h=face;boxes={"forehead":(x+.30*w,y+.12*h,x+.70*w,y+.30*h),"left_cheek":(x+.12*w,y+.48*h,x+.38*w,y+.72*h),"right_cheek":(x+.62*w,y+.48*h,x+.88*w,y+.72*h)};out={}
    for name,(x1,y1,x2,y2) in boxes.items():
        r=frame[max(0,int(y1)):min(frame.shape[0],int(y2)),max(0,int(x1)):min(frame.shape[1],int(x2))]
        if not r.size:return None
        ycc=cv2.cvtColor(r,cv2.COLOR_BGR2YCrCb);mask=(ycc[:,:,1]>=133)&(ycc[:,:,1]<=173)&(ycc[:,:,2]>=77)&(ycc[:,:,2]<=127)
        pixels=r[mask] if mask.sum()>=max(20,r.shape[0]*r.shape[1]*.08) else r.reshape(-1,3)
        out[name]={"rgb":np.mean(pixels[:,::-1],axis=0),"skin_pixels":int(mask.sum())}
    return out
def _region_candidate(rgb,fs):
    z=_filter(_pos(rgb),fs,HR_LOW,HR_HIGH)
    if z is None:return None
    f,p=welch(z,fs=fs,nperseg=len(z));keep=(f>=HR_LOW)&(f<=HR_HIGH);f,p=f[keep],p[keep]
    if not len(p):return None
    peaks,props=find_peaks(p,prominence=max(p)*.05);i=peaks[np.argmax(p[peaks])] if len(peaks) else np.argmax(p)
    return {"bpm":float(f[i]*60),"quality":float(p[i]/max(p.mean(),1e-9)),"prominence":float(props.get("prominences",[0])[np.argmax(p[peaks])] if len(peaks) else 0),"wave":z,"frequency":f,"power":p}
def _hr(regions,times,fs):
    """Fuse normalized per-ROI PSDs; 12s windows imply a documented ~5 BPM true resolution."""
    times=np.asarray(times);grid=np.arange(times[0],times[-1],1/fs);signals={n:np.column_stack([np.interp(grid,times,np.asarray(v)[:,c]) for c in range(3)]) for n,v in regions.items()};size=int(12*fs);tol=max(5.0,60/(size/fs));records=[];stats={n:[] for n in signals}
    for start in range(int(3*fs),max(int(3*fs),len(grid)-int(2*fs)-size+1),max(1,int(fs))):
        cand={n:_region_candidate(v[start:start+size],fs) for n,v in signals.items()};valid={n:x for n,x in cand.items() if x and x["quality"]>=3 and x["prominence"]>0}
        for n,x in valid.items():stats[n].append(x)
        fused=None;support=[]
        if valid:
            f=next(iter(valid.values()))["frequency"];fused_power=np.zeros_like(f);weights={}
            for n,x in valid.items():
                weights[n]=x["quality"]*(1+x["prominence"]/max(np.max(x["power"]),1e-9));fused_power+=weights[n]*(x["power"]/max(np.max(x["power"]),1e-9))
            peaks,props=find_peaks(fused_power,prominence=max(fused_power)*.05);i=peaks[np.argmax(fused_power[peaks])] if len(peaks) else np.argmax(fused_power);fused={"bpm":float(f[i]*60),"quality":float(fused_power[i]/max(fused_power.mean(),1e-9)),"prominence":float(props.get("prominences",[0])[np.argmax(fused_power[peaks])] if len(peaks) else 0)};support=[n for n,x in valid.items() if abs(x["bpm"]-fused["bpm"])<=tol]
        pairs={"forehead_left_difference":abs(cand["forehead"]["bpm"]-cand["left_cheek"]["bpm"]) if cand["forehead"] and cand["left_cheek"] else None,"forehead_right_difference":abs(cand["forehead"]["bpm"]-cand["right_cheek"]["bpm"]) if cand["forehead"] and cand["right_cheek"] else None,"left_right_difference":abs(cand["left_cheek"]["bpm"]-cand["right_cheek"]["bpm"]) if cand["left_cheek"] and cand["right_cheek"] else None}
        reason="two_or_more_supporting_rois" if len(support)>=2 else "fewer_than_two_supporting_rois";records.append({"center":start/fs+6,"cand":cand,"fused":fused,"support":support,"reason":reason,**pairs})
    raw=np.array([r["fused"]["bpm"] for r in records if r["fused"] and len(r["support"])>=2]);med=float(np.median(raw)) if len(raw) else None;mad=float(np.median(np.abs(raw-med))) if len(raw) else 0
    accepted=[r for r in records if r["fused"] and len(r["support"])>=2 and (mad==0 or abs(r["fused"]["bpm"]-med)<=max(5,3*mad))];vals=np.array([r["fused"]["bpm"] for r in accepted]);pct=len(vals)/max(1,len(records))*100;sd=float(np.std(vals)) if len(vals) else None;quality="GOOD" if len(vals) and pct>=60 and sd<=5 else "MODERATE" if len(vals) and pct>=40 and sd<=8 else "LOW";wave=_filter(_pos(signals["forehead"]),fs,HR_LOW,HR_HIGH)
    DEBUG=BASE_DIR/"results"/"debug";DEBUG.mkdir(parents=True,exist_ok=True)
    accepted_ids={id(r) for r in accepted}
    with (DEBUG/"hr_windows.csv").open("w",newline="",encoding="utf-8") as file:
        fields=["window_center_sec","forehead_hr","forehead_quality","forehead_peak_prominence","left_cheek_hr","left_cheek_quality","left_cheek_peak_prominence","right_cheek_hr","right_cheek_quality","right_cheek_peak_prominence","forehead_left_difference","forehead_right_difference","left_right_difference","consensus_hr","regions_used","accepted","rejection_reason"];w=csv.DictWriter(file,fieldnames=fields);w.writeheader()
        for r in records:
            row={"window_center_sec":r["center"],"consensus_hr":r["fused"]["bpm"] if r["fused"] else None,"regions_used":",".join(r["support"]),"accepted":id(r) in accepted_ids,"rejection_reason":r["reason"],"forehead_left_difference":r["forehead_left_difference"],"forehead_right_difference":r["forehead_right_difference"],"left_right_difference":r["left_right_difference"]}
            for n in stats:
                x=r["cand"][n];row[f"{n}_hr"]=x["bpm"] if x else None;row[f"{n}_quality"]=x["quality"] if x else None;row[f"{n}_peak_prominence"]=x["prominence"] if x else None
            w.writerow(row)
    summary={n:{"candidate_windows":len(v),"median_hr":round(float(np.median([x["bpm"] for x in v])),2) if v else None,"median_spectral_quality":round(float(np.median([x["quality"] for x in v])),2) if v else None,"reliability_score":round(100*len(v)/max(1,len(records))*min(1,np.median([x["quality"] for x in v])/5),1) if v else 0} for n,v in stats.items()};diffs=[abs(r["cand"][a]["bpm"]-r["cand"][b]["bpm"]) for r in accepted for a,b in [("forehead","left_cheek"),("forehead","right_cheek"),("left_cheek","right_cheek")] if a in r["support"] and b in r["support"]]
    diag={"effective_sampling_rate":round(fs,2),"hr_total_windows":len(records),"hr_fused_candidate_windows":sum(r["fused"] is not None for r in records),"hr_accepted_windows":len(vals),"hr_accepted_percent":round(pct,1),"hr_candidate_median":round(med,2) if med else None,"final_hr_median":round(float(np.median(vals)),2) if len(vals) else None,"final_hr_sd":round(sd,2) if sd is not None else None,"two_region_consensus_windows":sum(len(r["support"])==2 for r in records),"three_region_consensus_windows":sum(len(r["support"])==3 for r in records),"no_consensus_windows":sum(len(r["support"])<2 for r in records),"median_roi_difference_bpm":round(float(np.median(diffs)),2) if diffs else None,"median_agreeing_regions":round(float(np.median([len(r["support"]) for r in accepted])),1) if accepted else 0,"fused_median_peak_quality":round(float(np.median([r["fused"]["quality"] for r in accepted])),2) if accepted else None,"regions":summary,"decision":"ACCEPTED" if quality!="LOW" else "REJECTED","decision_reasons":["insufficient temporal coverage"] if quality=="LOW" else []}
    return (float(np.median(vals)) if quality!="LOW" and len(vals) else None),quality,len(vals),wave,diag
def _hrv(wave,fs,quality):
    if wave is None or quality!="GOOD":return None,None,0,"LOW"
    p,_=find_peaks(wave,distance=max(1,int(fs*.33)),prominence=max(np.std(wave)*.35,1e-6));ibi=np.diff(p)/fs;ibi=ibi[(ibi>=.35)&(ibi<=1.5)]
    if len(ibi):ibi=ibi[np.abs(ibi-np.median(ibi))<=max(.15,np.median(ibi)*.2)]
    if len(ibi)<20:return None,None,len(ibi),"LOW"
    return float(np.sqrt(np.mean(np.diff(ibi)**2))*1000),float(np.std(ibi,ddof=1)*1000),len(ibi),"GOOD" if len(ibi)>=30 else "MODERATE"
def _rr(frames,roi,fs):
    if roi is None or len(frames)<fs*15:return None,"LOW"
    x,y,w,h=roi;prev=cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY);mask=np.zeros_like(prev);mask[y:y+h,x:x+w]=255;pts=cv2.goodFeaturesToTrack(prev,mask=mask,maxCorners=100,qualityLevel=.02,minDistance=8)
    if pts is None:return None,"LOW"
    moves=[]
    for frame in frames[1:]:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);nxt,status,_=cv2.calcOpticalFlowPyrLK(prev,gray,pts,None,winSize=(15,15),maxLevel=2)
        if nxt is not None and status is not None:
            old=pts.reshape(-1,2)[status.ravel()==1];new=nxt.reshape(-1,2)[status.ravel()==1]
            if len(new)>=8:moves.append(float(np.median(new[:,1]-old[:,1])));pts=new.reshape(-1,1,2).astype(np.float32)
            else:nxt=None
        if nxt is None:
            pts=cv2.goodFeaturesToTrack(gray,mask=mask,maxCorners=100,qualityLevel=.02,minDistance=8)
            if pts is None:break
        prev=gray
    z=_filter(np.cumsum(moves),fs,RR_LOW,RR_HIGH)
    if z is None:return None,"LOW"
    f,p=welch(z,fs=fs,nperseg=len(z),nfft=8192);keep=(f>=RR_LOW)&(f<=RR_HIGH);f,p=f[keep],p[keep];i=np.argmax(p);psd=f[i]*60;ratio=p[i]/max(p.mean(),1e-9);ac=np.correlate(z,z,"full")[len(z)-1:];lo,hi=int(fs/RR_HIGH),min(len(ac)-1,int(fs/RR_LOW));auto=60/((lo+np.argmax(ac[lo:hi+1]))/fs);d=abs(psd-auto);q="GOOD" if d<=2.5 and ratio>=3 else "MODERATE" if d<=4 and ratio>=2.5 else "LOW";return (float((psd+auto)/2) if q!="LOW" else None),q
def analyze_video(video_path,force_reanalyze=False,progress_callback=None):
    start=time.perf_counter();source=Path(video_path).resolve();CACHE_DIR.mkdir(parents=True,exist_ok=True);k=_key(source);result_path=CACHE_DIR/f"{source.stem}_{k}_analysis.json"
    if result_path.exists() and not force_reanalyze:
        _notify(progress_callback,"Loading cached analysis",100)
        with result_path.open(encoding="utf-8") as f:return json.load(f)
    if not CASCADE_PATH.exists():return _empty([f"Missing Haar cascade: {CASCADE_PATH}"])
    # Decode the original once, skip source frames, and resize in memory. Never re-encode rPPG input.
    preparation=time.perf_counter()
    cap,decode_info=_open_capture(source)
    preparation=time.perf_counter()-preparation
    _notify(progress_callback,"Preparing video",100)
    if not cap.isOpened():return _empty(["Could not open the selected video."])
    fps=cap.get(cv2.CAP_PROP_FPS) or TARGET_ANALYSIS_FPS;total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));source_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));source_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));duration=total/fps if total else 0;result=_empty([],duration);result.update(video_supported=True,video_processing_status="SUCCESS",source_width=source_width,source_height=source_height)
    if duration<15:cap.release();return _empty(["Video is too short; use at least 15 seconds."],duration)
    cascade=cv2.CascadeClassifier(str(CASCADE_PATH));extract=time.perf_counter();step=max(1,round(fps/TARGET_ANALYSIS_FPS));fs=fps/step;regions={"forehead":[],"left_cheek":[],"right_cheek":[]};times=[];skin={"forehead":[],"left_cheek":[],"right_cheek":[]};frames=[];face=chest=None;usable=processed=index=0;decode_seconds=resize_seconds=face_seconds=rgb_seconds=motion_seconds=0.0;face_calls=0;analysis_width=RPPG_ANALYSIS_WIDTH;face_widths=[];face_heights=[];adapted=False
    while True:
        t=time.perf_counter()
        ok,frame=cap.read()
        decode_seconds+=time.perf_counter()-t
        if not ok:break
        index+=1
        timestamp_seconds=(index-1)/fps
        if timestamp_seconds>=MAX_ANALYSIS_SECONDS:break
        if index%step:continue
        processed+=1
        if frame.shape[1]!=analysis_width:
            t=time.perf_counter();s=analysis_width/frame.shape[1];frame=cv2.resize(frame,(analysis_width,max(1,int(frame.shape[0]*s))),interpolation=cv2.INTER_AREA);resize_seconds+=time.perf_counter()-t
        if face is None or processed%FACE_DETECTION_INTERVAL==1:
            t=time.perf_counter();faces=cascade.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),1.1,6,minSize=(45,45));face_seconds+=time.perf_counter()-t;face_calls+=1
            if len(faces):
                face=max(faces,key=lambda b:b[2]*b[3]);face_widths.append(int(face[2]));face_heights.append(int(face[3]))
                # One deterministic increase preserves facial ROI detail when a distant face is too small at 800px.
                if not adapted and face[2] < MIN_FACE_WIDTH_PIXELS and source_width > RPPG_ANALYSIS_WIDTH:
                    analysis_width=RPPG_LARGE_FACE_WIDTH;adapted=True;face=None
        if face is not None:
            t=time.perf_counter();v=_roi(frame,face);rgb_seconds+=time.perf_counter()-t
            if v is not None:
                t=time.perf_counter();usable+=1;times.append(timestamp_seconds);[regions[n].append(v[n]["rgb"]) for n in regions];[skin[n].append(v[n]["skin_pixels"]) for n in skin];x,y,w,h=map(int,face);scale=MOTION_WIDTH/frame.shape[1];motion=cv2.resize(frame,(MOTION_WIDTH,max(1,int(frame.shape[0]*scale))),interpolation=cv2.INTER_AREA);chest=(max(0,int((x-.15*w)*scale)),min(motion.shape[0]-1,int((y+.82*h)*scale)),min(motion.shape[1],int(1.3*w*scale)),min(motion.shape[0]-int((y+.82*h)*scale),int(.9*h*scale)));frames.append(motion);motion_seconds+=time.perf_counter()-t
        if total and processed%30==0:_notify(progress_callback,"Extracting camera signals",int(index/total*100))
    cap.release();extract=time.perf_counter()-extract;result["face_detection_percent"]=round(100*usable/max(1,processed),1);analysed_duration=min(duration,MAX_ANALYSIS_SECONDS);analysis_height=round(source_height*analysis_width/source_width);median_width=float(np.median(face_widths)) if face_widths else 0;median_height=float(np.median(face_heights)) if face_heights else 0;median_area=100*median_width*median_height/max(1,analysis_width*analysis_height);result.update(source_duration_seconds=round(duration,2),analysed_duration_seconds=round(analysed_duration,2));extraction_details={**decode_info,"source_width":source_width,"source_height":source_height,"source_fps":round(fps,2),"source_orientation":"landscape" if source_width>=source_height else "portrait","source_aspect_ratio":round(source_width/max(1,source_height),3),"analysis_width":analysis_width,"analysis_height":analysis_height,"median_face_width":round(median_width,1),"median_face_height":round(median_height,1),"median_face_area_percent":round(median_area,2),"source_duration_seconds":round(duration,2),"analysed_duration_seconds":round(analysed_duration,2),"effective_sampling_rate":round(fs,2),"video_decode_seconds":round(decode_seconds,2),"resize_seconds":round(resize_seconds,2),"face_detection_seconds":round(face_seconds,2),"rgb_extraction_seconds":round(rgb_seconds,2),"motion_extraction_seconds":round(motion_seconds,2),"analysed_frames":processed,"face_detection_calls":face_calls}
    if len(times)<fs*15:result["warnings"].append("Too few usable face frames for rPPG analysis.");result["timing"]={"preparing_video_seconds":round(preparation,2),"video_extraction_seconds":round(extract,2),**extraction_details,"total_seconds":round(time.perf_counter()-start,2)};return result
    _notify(progress_callback,"Calculating heart rate",100);t=time.perf_counter();hr,hrq,accepted,wave,hr_debug=_hr(regions,times,fs);hrt=time.perf_counter()-t
    _notify(progress_callback,"Calculating HRV",100);t=time.perf_counter();rmssd,sdnn,beats,hrvq=_hrv(wave,fs,hrq);hrvt=time.perf_counter()-t
    _notify(progress_callback,"Calculating respiratory rate",100);t=time.perf_counter();rr,rrq=_rr(frames,chest,fs);rrt=time.perf_counter()-t
    hr_debug.update(mean_skin_pixels_forehead=round(float(np.mean(skin["forehead"])),1),mean_skin_pixels_left_cheek=round(float(np.mean(skin["left_cheek"])),1),mean_skin_pixels_right_cheek=round(float(np.mean(skin["right_cheek"])),1))
    result.update(heart_rate=round(hr,1) if hr is not None else None,heart_rate_candidate=hr_debug["hr_candidate_median"],heart_rate_quality=hrq,respiratory_rate=round(rr,1) if rr is not None else None,respiratory_quality=rrq,rmssd=round(rmssd,1) if rmssd is not None else None,sdnn=round(sdnn,1) if sdnn is not None else None,hrv_quality=hrvq,accepted_beats=beats,hr_diagnostics=hr_debug)
    if hr is None:result["warnings"].append("Insufficient rPPG signal for a reliable heart-rate result.")
    if rr is None:result["warnings"].append("Insufficient shoulder-motion signal for a reliable respiratory-rate result.")
    score=.35*result["face_detection_percent"]+.35*min(100,accepted*4)+.2*(100 if hrq=="GOOD" else 60 if hrq=="MODERATE" else 0)+.1*(100 if rrq=="GOOD" else 60 if rrq=="MODERATE" else 0);result["signal_quality_score"]=round(score);result["signal_quality_label"]="GOOD" if score>=85 else "MODERATE" if score>=65 else "LOW";result["timing"]={"preparing_video_seconds":round(preparation,2),"video_extraction_seconds":round(extract,2),**extraction_details,"heart_rate_seconds":round(hrt,2),"hrv_seconds":round(hrvt,2),"respiratory_seconds":round(rrt,2),"total_seconds":round(time.perf_counter()-start,2)}
    with result_path.open("w",encoding="utf-8") as f:json.dump(result,f,indent=2)
    _notify(progress_callback,"Analysis complete",100);return result
