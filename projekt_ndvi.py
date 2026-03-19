import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from os import makedirs, listdir
from pathlib import Path
from constants import *

def equalize_hsv(img):
    """Wyrównanie histogramu w przestrzeni HSV (tylko kanał V)."""
    hsv_img = img.convert('HSV')
    h, s, v = hsv_img.split()
    v_eq = ImageOps.equalize(v)
    return Image.merge('HSV', (h, s, v_eq)).convert('RGB')

def threshold_image(array, threshold_val):
    """Progowanie indeksu (1=las, 0=brak)."""
    return (array > threshold_val).astype(float)

def calculate_gli(arr):
    return np.nan_to_num((2*arr[:,:,1]-arr[:,:,0]-arr[:,:,2])/(2*arr[:,:,1]+arr[:,:,0]+arr[:,:,2]))

def calculate_vari(arr):
    return np.nan_to_num((arr[:,:,1]-arr[:,:,0])/(arr[:,:,1]+arr[:,:,0]-arr[:,:,2]))

def calculate_vigreen(arr):
    return np.nan_to_num((arr[:,:,1]-arr[:,:,0])/(arr[:,:,1]+arr[:,:,0]))

def save_index_image(array, save_path, index_name, color_map, vmin, vmax):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    cim = ax.imshow(array, vmin=vmin, vmax=vmax, cmap=color_map, aspect="auto")
    
    cbar = fig.colorbar(cim)
    cbar.set_label(f"Wartość indeksu {index_name}")
    
    ax.set_title(f"Indeks {index_name}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def run_analysis():
    analysis_data = {}
    img_files = [f for f in listdir(INPUT_IMAGES_PATH) if Path(f).suffix.lower() in IMG_EXTENSIONS]
    img_files.sort()

    for img_name in img_files:
        img_path = Path(INPUT_IMAGES_PATH) / img_name
        year = "".join(filter(str.isdigit, img_name)) or img_name
        analysis_data[year] = {}

        # 1. Histogram Equalization (Wyrównanie HSV dla wizualizacji)
        img_raw = Image.open(img_path).convert('RGB')
        img_eq = equalize_hsv(img_raw)
        img_eq.save(Path(OUTPUT_IMAGES_PATH) / f"{img_path.stem}_equalized.png")

        arr = np.asarray(img_raw).astype(float)

        for idx_name in ["VARI", "GLI", "VI"]:
            if idx_name == "VARI": 
                array = calculate_vari(arr)
                gray_dir, col_dir = VARI_IMAGES_GRAY, VARI_IMAGES_COLOR
                threshold_val = 0.05
            elif idx_name == "GLI": 
                array = calculate_gli(arr)
                gray_dir, col_dir = GLI_IMAGES_GRAY, GLI_IMAGES_COLOR
                threshold_val = 0.1
            else: # VI / VIGREEN
                array = calculate_vigreen(arr)
                gray_dir, col_dir = VI_IMAGES_GRAY, VI_IMAGES_COLOR
                threshold_val = 0.02

            vmin = np.percentile(array, 10)
            vmax = np.percentile(array, 90)

            save_index_image(array, Path(gray_dir) / f"{img_path.stem}_gray.png", idx_name, "gray", vmin, vmax)
            save_index_image(array, Path(col_dir) / f"{img_path.stem}_color.png", idx_name, "RdYlGn", vmin, vmax)

            binary_mask = threshold_image(array, threshold_val)
            plt.imsave(Path(OUTPUT_IMAGES_PATH) / f"{img_path.stem}_{idx_name}_mask.png", binary_mask, cmap='binary')

            forest_percent = (np.sum(binary_mask) / binary_mask.size) * 100
            analysis_data[year][idx_name] = forest_percent


    generate_summary_plot(analysis_data)

def generate_summary_plot(data):
    years = sorted(data.keys())
    plt.figure(figsize=(10, 6))
    
    labels_map = {"VARI": "VARI", "GLI": "GLI", "VI": "VIGREEN"}
    
    for idx_key, label in labels_map.items():
        values = [data[y][idx_key] for y in years]
        plt.plot(years, values, marker='o', label=label)

    plt.title("Percentage of Forest Area Over Time")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Forest Area (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(Path(OUTPUT_IMAGES_PATH) / "deforestation_trend.png")
    plt.show()

def prepare_folders(*paths):
    for p in paths: makedirs(p, exist_ok=True)

if __name__ == "__main__":
    prepare_folders(
        INPUT_IMAGES_PATH, OUTPUT_IMAGES_PATH, 
        VARI_IMAGES_GRAY, VARI_IMAGES_COLOR, 
        GLI_IMAGES_GRAY, GLI_IMAGES_COLOR, 
        VI_IMAGES_GRAY, VI_IMAGES_COLOR
    )
    np.seterr(divide='ignore', invalid='ignore')
    run_analysis()
    print("Gotowe! Sprawdź folder 'output images'.")