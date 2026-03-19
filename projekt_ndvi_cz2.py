import numpy as np
import matplotlib.pyplot as plt
import rasterio
from os import makedirs, listdir
from pathlib import Path
from constants import *

def calculate_ndvi(ds):
 
    red = ds.read(3).astype(float)
    nir = ds.read(4).astype(float)
    
    denominator = nir + red
    ndvi = np.divide((nir - red), denominator, out=np.zeros_like(nir), where=denominator != 0)
    return ndvi

def save_ndvi_map(ndvi_array, save_path, year):
    """Generuje mapę NDVI z paskiem kolorów."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    cim = ax.imshow(ndvi_array, vmin=-1, vmax=1, cmap='RdYlGn')
    
    cbar = fig.colorbar(cim)
    cbar.set_label('Wskaźnik NDVI')
    
    ax.set_title(f"Mapa NDVI - Rok {year}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def save_binary_vegetation(ndvi_array, save_path, threshold=0.2):
    binary = (ndvi_array > threshold).astype(float)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(binary, cmap='gray')
    plt.title(f"Klasyfikacja: Roślinność (NDVI > {threshold})")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def process_tiffs():
    ndvi_stats = {}
    
    tiff_files = [f for f in listdir(INPUT_IMAGES_PATH) if Path(f).suffix.lower() in [".tif", ".tiff"]]
    tiff_files.sort()

    for tiff_name in tiff_files:
        tiff_path = Path(INPUT_IMAGES_PATH) / tiff_name
        year = "".join(filter(str.isdigit, tiff_name)) or tiff_name
        
        with rasterio.open(tiff_path) as src:
            ndvi = calculate_ndvi(src)
            
            save_ndvi_map(ndvi, Path(OUTPUT_IMAGES_PATH) / f"{tiff_path.stem}_NDVI.png", year)
            
            save_binary_vegetation(ndvi, Path(OUTPUT_IMAGES_PATH) / f"{tiff_path.stem}_binary.png", threshold=0.3)
            
            ndvi_stats[year] = np.mean(ndvi)

    generate_ndvi_trend(ndvi_stats)

def generate_ndvi_trend(stats):
    years = sorted(stats.keys())
    values = [stats[y] for y in years]
    
    plt.figure(figsize=(12, 6), dpi=100)
    
    plt.bar(years, values, color="#074672", edgecolor='grey', width=0.6)
    
    plt.title("Deforestation Analysis", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage of Forest (%)", fontsize=12)
    plt.ylim(0, 100) 
    plt.grid(True, axis='y', linestyle='--', alpha=0.7) 
    plt.xticks(rotation=45) 
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_IMAGES_PATH) / "deforestation_summary.png")
    plt.show()

def process_tiffs():
    ndvi_stats = {} 
    
    tiff_files = [f for f in listdir(INPUT_IMAGES_PATH) if Path(f).suffix.lower() in [".tif", ".tiff"]]
    tiff_files.sort()

    for tiff_name in tiff_files:
        tiff_path = Path(INPUT_IMAGES_PATH) / tiff_name
        
        year_label = tiff_path.stem 
        
        with rasterio.open(tiff_path) as src:
            ndvi = calculate_ndvi(src)
            
            save_ndvi_map(ndvi, Path(OUTPUT_IMAGES_PATH) / f"{tiff_path.stem}_NDVI.png", year_label)
            save_binary_vegetation(ndvi, Path(OUTPUT_IMAGES_PATH) / f"{tiff_path.stem}_binary.png")
            forest_mask = (ndvi > 0.3).astype(int)
            forest_percentage = (np.sum(forest_mask) / forest_mask.size) * 100           
            ndvi_stats[year_label] = forest_percentage

    if ndvi_stats:
        generate_ndvi_trend(ndvi_stats)
    else:
        print("Nie znaleziono plików TIFF do analizy.")

if __name__ == "__main__":
    makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)
    np.seterr(divide='ignore', invalid='ignore')
    process_tiffs()