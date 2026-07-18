import os
from PIL import Image

def extract():
    src_dir = "outputs/week04_multimodal/ket_qua_kiem_tra"
    dest_dir = "datasets/sample_images"
    os.makedirs(dest_dir, exist_ok=True)
    
    files = {
        "broken_bottle_sample.png": "w4_broken_large_ket_qua_kiem_tra_tieng_viet.png",
        "good_bottle_sample.png": "w4_good_ket_qua_kiem_tra_tieng_viet.png"
    }
    
    for dest_name, src_name in files.items():
        src_path = os.path.join(src_dir, src_name)
        dest_path = os.path.join(dest_dir, dest_name)
        
        if os.path.exists(src_path):
            img = Image.open(src_path)
            print(f"Loaded {src_name} with size {img.size}")
            # The canvas height is panel_height + 42. Let's compute panel_width.
            # canvas width = panel_width * 3 + 24
            canvas_width, canvas_height = img.size
            panel_width = (canvas_width - 24) // 3
            panel_height = canvas_height - 42
            print(f"Calculated panel size: {panel_width}x{panel_height}")
            
            # Crop the first panel: x from 0 to panel_width, y from 42 to canvas_height
            cropped = img.crop((0, 42, panel_width, canvas_height))
            cropped.save(dest_path)
            print(f"Saved cropped image to {dest_path}")
        else:
            print(f"Source file {src_path} does not exist!")

if __name__ == "__main__":
    extract()
