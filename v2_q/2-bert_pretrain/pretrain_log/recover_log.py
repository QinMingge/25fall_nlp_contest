import json
import os
import datetime
import glob
import bisect

def recover_log(checkpoint_dir, output_log_file):
    # 1. Find the latest checkpoint to get the full history
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        print("No checkpoints found!")
        return

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = checkpoints[-1]
    print(f"Latest checkpoint: {latest_checkpoint}")

    # 2. Read trainer_state.json
    state_file = os.path.join(latest_checkpoint, "trainer_state.json")
    if not os.path.exists(state_file):
        print(f"trainer_state.json not found in {latest_checkpoint}")
        return

    with open(state_file, 'r') as f:
        data = json.load(f)
    
    log_history = data.get("log_history", [])
    if not log_history:
        print("No log history found.")
        return

    # 3. Build a map of step -> timestamp from checkpoint modification times
    # We use the modification time of the directory as the timestamp for that step
    step_time_map = {}
    
    # Also add start time if possible? 
    # We can assume the first checkpoint's speed applies backwards, or try to find 'runs' folder creation time.
    # For now, let's just get all checkpoint times.
    
    print("Gathering checkpoint timestamps...")
    for cp in checkpoints:
        try:
            step = int(cp.split("-")[-1])
            # Get modification time
            mtime = os.path.getmtime(cp)
            step_time_map[step] = mtime
        except ValueError:
            continue

    sorted_steps = sorted(step_time_map.keys())
    if not sorted_steps:
        print("No valid checkpoints found.")
        return

    # 4. Interpolate timestamps for each log entry
    # We need to assign a timestamp to every entry in log_history
    
    # Create a list of (step, timestamp) for interpolation
    known_points = [(step, step_time_map[step]) for step in sorted_steps]
    
    # If we have at least 2 points, we can calculate speed.
    # If we only have 1 point, we can't really know the start time unless we guess.
    # Let's assume a constant speed based on the first available interval if needed.
    
    with open(output_log_file, 'w') as f_out:
        for entry in log_history:
            step = entry.get("step")
            if step is None:
                continue
                
            # Find timestamp
            if step in step_time_map:
                ts = step_time_map[step]
            else:
                # Interpolate
                # Find insertion point
                idx = bisect.bisect_right(sorted_steps, step)
                
                if idx == 0:
                    # Before the first checkpoint. 
                    # We need to extrapolate backwards using the speed of the first interval.
                    if len(sorted_steps) >= 2:
                        step0 = sorted_steps[0]
                        step1 = sorted_steps[1]
                        time0 = step_time_map[step0]
                        time1 = step_time_map[step1]
                        
                        speed = (time1 - time0) / (step1 - step0) # seconds per step
                        
                        # time = time0 - (step0 - step) * speed
                        ts = time0 - (step0 - step) * speed
                    else:
                        # Only one checkpoint. Assume some arbitrary speed or just use that time (bad).
                        # Let's assume 1 sec per step if we have absolutely no info, but that's unlikely for BERT.
                        # Or better, check if there is a 'runs' directory and use its creation time as step 0.
                        # For now, let's just assume the speed is constant from step 0 to step1, 
                        # and assume step 0 started X seconds ago? 
                        # Actually, if we only have 1 checkpoint, we can't know the speed.
                        # But the user has many checkpoints.
                        ts = step_time_map[sorted_steps[0]] # Fallback
                        
                elif idx == len(sorted_steps):
                    # After the last checkpoint (shouldn't happen if we read from last checkpoint state)
                    # Extrapolate forward
                    step_last = sorted_steps[-1]
                    step_prev = sorted_steps[-2]
                    time_last = step_time_map[step_last]
                    time_prev = step_time_map[step_prev]
                    speed = (time_last - time_prev) / (step_last - step_prev)
                    ts = time_last + (step - step_last) * speed
                else:
                    # Between two checkpoints
                    step_after = sorted_steps[idx]
                    step_before = sorted_steps[idx-1]
                    time_after = step_time_map[step_after]
                    time_before = step_time_map[step_before]
                    
                    ratio = (step - step_before) / (step_after - step_before)
                    ts = time_before + ratio * (time_after - time_before)

            # Format timestamp
            dt_object = datetime.datetime.fromtimestamp(ts)
            time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")
            
            # Construct log message
            # Format: YYYY-MM-DD HH:MM:SS - INFO - Epoch: X, Step: Y, Train Loss: Z, ...
            
            msg_parts = []
            if 'epoch' in entry:
                msg_parts.append(f"Epoch: {entry['epoch']:.2f}")
            
            msg_parts.append(f"Step: {step}")
            
            if 'loss' in entry:
                msg_parts.append(f"Train Loss: {entry['loss']:.4f}")
            
            if 'eval_loss' in entry:
                msg_parts.append(f"Eval Loss: {entry['eval_loss']:.4f}")
            if 'eval_accuracy' in entry:
                msg_parts.append(f"Accuracy: {entry['eval_accuracy']:.4f}")
            if 'eval_f1_macro' in entry:
                msg_parts.append(f"F1: {entry['eval_f1_macro']:.4f}")
            if 'learning_rate' in entry:
                msg_parts.append(f"Learning Rate: {entry['learning_rate']:.2e}")

            log_line = f"{time_str} - INFO - {', '.join(msg_parts)}\n"
            f_out.write(log_line)

    print(f"Successfully recovered log to {output_log_file}")

if __name__ == "__main__":
    checkpoint_dir = "/data/jinda/qinmingge/BERT/25fallnewsclassify/output_small_4096"
    output_log_file = "recovered_pretrain.log"
    recover_log(checkpoint_dir, output_log_file)
