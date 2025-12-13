import json
import os
import datetime
import glob
import bisect

def recover_cnn_log(checkpoint_dir, output_log_file):
    # 1. Find the latest checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        print("No checkpoints found!")
        return

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

    # 3. Determine timestamps
    # Since we don't have many checkpoints (only 9000 and 10500?), we might need to rely on the console log start time.
    # Console log says start time: 12/12/2025 23:35:28
    # Let's use that as T0.
    
    start_time_str = "2025-12-12 23:35:28"
    start_dt = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    start_ts = start_dt.timestamp()

    # We need to estimate speed.
    # Let's check the modification time of the latest checkpoint to get the end time (or current time).
    latest_mtime = os.path.getmtime(latest_checkpoint)
    latest_step = int(latest_checkpoint.split("-")[-1])
    
    # Calculate average seconds per step
    # But wait, did it run continuously?
    # Assuming yes.
    
    total_seconds = latest_mtime - start_ts
    if total_seconds <= 0:
        # Fallback if file times are weird (e.g. copied files)
        # Let's assume 0.5s per step?
        # From console log: eval_runtime': 183.3248.
        # Training is usually faster per step than eval per step? No, batch size differs.
        # Let's try to use the checkpoints we have.
        pass
    else:
        avg_sec_per_step = total_seconds / latest_step
    
    # Refined approach:
    # Use the checkpoints we have to build a time map, anchored at start_time for step 0.
    step_time_map = {0: start_ts}
    
    for cp in checkpoints:
        try:
            step = int(cp.split("-")[-1])
            mtime = os.path.getmtime(cp)
            step_time_map[step] = mtime
        except ValueError:
            continue
            
    sorted_steps = sorted(step_time_map.keys())
    
    # 4. Write log
    with open(output_log_file, 'w') as f_out:
        for entry in log_history:
            step = entry.get("step")
            if step is None:
                continue
                
            # Interpolate
            idx = bisect.bisect_right(sorted_steps, step)
            if idx == 0:
                # Should be covered by step 0 anchor
                ts = start_ts
            elif idx == len(sorted_steps):
                # Extrapolate
                step_last = sorted_steps[-1]
                step_prev = sorted_steps[-2]
                time_last = step_time_map[step_last]
                time_prev = step_time_map[step_prev]
                speed = (time_last - time_prev) / (step_last - step_prev)
                ts = time_last + (step - step_last) * speed
            else:
                step_after = sorted_steps[idx]
                step_before = sorted_steps[idx-1]
                time_after = step_time_map[step_after]
                time_before = step_time_map[step_before]
                ratio = (step - step_before) / (step_after - step_before)
                ts = time_before + ratio * (time_after - time_before)
            
            dt_object = datetime.datetime.fromtimestamp(ts)
            time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")
            
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

    print(f"Successfully recovered CNN log to {output_log_file}")

if __name__ == "__main__":
    checkpoint_dir = "/data/jinda/qinmingge/BERT/25fallnewsclassify/output_ft_cnn"
    output_log_file = "recovered_cnn_training.log"
    recover_cnn_log(checkpoint_dir, output_log_file)
