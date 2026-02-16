import os
import shutil

def copy_files():
    # Define the copy tasks
    tasks = [
        {
            "name": "Predicted Labels",
            "src": "/mnt/semkitti/kth_day_6_labels/inferred_labels_ALL/kth_day_06/inferred_labels/",
            "dst": "pred_labels"
        },
        {
            "name": "Ground Truth Labels",
            "src": "/mnt/semkitti/MCD-finalized-dataset/MCD-finalized-dataset/kth_day_06/gt_labels/",
            "dst": "gt_labels"
        },
        {
            "name": "LiDAR Data",
            "src": "/mnt/semkitti/MCD-finalized-dataset/MCD-finalized-dataset/kth_day_06/lidar_bin/data/",
            "dst": "data"
        }
    ]

    # Process each task
    for task in tasks:
        src_dir = task["src"]
        dst_dir = task["dst"]
        print(f"\n--- Processing {task['name']} ---")
        print(f"Source: {src_dir}")
        print(f"Dest:   {dst_dir}")

        # Create destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)

        count = 0
        # Iterate through indices 0 to 100
        for i in range(101):
            # Format index as 10-digit zero-padded string (standard KITTI format)
            file_id = f"{i:010d}"
            
            # Check for possible extensions (observed .bin in your folders, but .label is common for labels)
            found = False
            for ext in [".bin", ".label"]:
                filename = f"{file_id}{ext}"
                src_path = os.path.join(src_dir, filename)
                
                if os.path.exists(src_path):
                    dst_path = os.path.join(dst_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {filename}")
                    found = True
                    count += 1
                    break
            
            if not found:
                # Optional: uncomment to see missing files
                # print(f"Missing: {file_id} (checked .bin and .label)")
                pass

        print(f"Total files copied for {task['name']}: {count}")

if __name__ == "__main__":
    copy_files()