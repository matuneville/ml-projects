from PIL import Image
import numpy as np
import pandas as pd
import csv

def convert_image_to_csv(image_path, csv_path):
    # Load image and convert to greyscale
    img = Image.open(image_path).convert('L')  

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Save the numpy array to a CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in img_array:
            writer.writerow(row)

    print(f"Conversion complete! CSV file saved as {csv_path}")

# Example usage
# convert_image_to_csv('your_image.jpg', 'image_data.csv')

def csv2dataframe_clf(csv_path, pixel_columns, label):
    num = pd.read_csv(csv_path, header=None)
    num = pd.DataFrame(num.values.flatten()).T
    num.columns = pixel_columns
    num['label'] = label
    
    return num