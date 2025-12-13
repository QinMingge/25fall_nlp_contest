import json
import os
import datetime
import glob
import bisect

def recover_linear_log(checkpoint_dir, output_log_file):
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
    # We need a start time. 
    # Let's assume the user ran it recently or we can infer from file times.
    # If we don't have a console log start time, we can try to use the first checkpoint time minus some delta.
    # Or just use the checkpoint times for interpolation.
    
    # Let's try to find the start time from the first checkpoint creation time?
    # Or just rely purely on checkpoint modification times.
    
    step_time_map = {}
    for cp in checkpoints:
        try:
            step = int(cp.split("-")[-1])
            mtime = os.path.getmtime(cp)
            step_time_map[step] = mtime
        except ValueError:
            continue
            
    sorted_steps = sorted(step_time_map.keys())
    
    if not sorted_steps:
        print("No valid checkpoints found to determine time.")
        return

    # 4. Write log
    with open(output_log_file, 'w') as f_out:
        for entry in log_history:
            step = entry.get("step")
            if step is None:
                continue
                
            # Interpolate
            if step in step_time_map:
                ts = step_time_map[step]
            else:
                idx = bisect.bisect_right(sorted_steps, step)
                if idx == 0:
                    # Extrapolate backwards
                    if len(sorted_steps) >= 2:
                        step0 = sorted_steps[0]
                        step1 = sorted_steps[1]
                        time0 = step_time_map[step0]
                        time1 = step_time_map[step1]
                        speed = (time1 - time0) / (step1 - step0)
                        ts = time0 - (step0 - step) * speed
                    else:
                        # Only one checkpoint, can't know speed.
                        # Assume it finished just now?
                        ts = step_time_map[sorted_steps[0]]
                elif idx == len(sorted_steps):
                    # Extrapolate forwards
                    step_last = sorted_steps[-1]
                    step_prev = sorted_steps[-2]
                    time_last = step_time_map[step_last]
                    time_prev = step_time_map[step_prev]
                    speed = (time_last - time_prev) / (step_last - step_prev)
                    ts = time_last + (step - step_last) * speed
                else:
                    # Interpolate
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

    print(f"Successfully recovered Linear log to {output_log_file}")

if __name__ == "__main__":
    checkpoint_dir = "/data/jinda/qinmingge/BERT/25fallnewsclassify/output_ft_linear"
    output_log_file = "recovered_linear_training.log"
    recover_linear_log(checkpoint_dir, output_log_file)
