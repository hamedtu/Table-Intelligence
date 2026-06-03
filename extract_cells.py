
"""Main script for extracting table cells from images using OpenCV.

Acknowledges and Authorship Note:
This script was developed with AI code assistance (GitHub Copilot) for initial scaffolding and suggestions,
but all core function workflow and logic were manually reviewed, tested, and refined by the author [Hamed Kelardeh].

"""



import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm


@dataclass
class BoxAnnotation:
    x: int  # upper left corner x 
    y: int  # upper left corner y 
    width: int  # box width 
    height: int  # box height 
    class_name: str  # identifier for row and column starting at 0 


@dataclass
class TableConfig:
    """Configuration parameters for table cell extraction.
    
    All parameters are tunable to adapt to different table types and image qualities.
    """
    # Image preprocessing
    binary_threshold: int = 128  # Threshold for binary conversion (0-255)
    
    # Line detection - kernel configuration
    kernel_length_divisor: int = 100  # Image width divided by this = kernel length
    vertical_kernel_width: int = 1  # Width of vertical line detection kernel
    horizontal_kernel_height: int = 1  # Height of horizontal line detection kernel
    
    # Morphological operations
    erode_iterations: int = 3  # Number of erosion iterations for line detection
    dilate_iterations: int = 3  # Number of dilation iterations for line detection
    final_erode_iterations: int = 2  # Final erosion after combining lines
    
    # Line combination weights
    vertical_line_weight: float = 0.5  # Weight for vertical lines (0-1)
    horizontal_line_weight: float = 0.5  # Weight for horizontal lines (0-1)
    
    # Cell filtering - size constraints
    min_cell_width: int = 10  # Minimum cell width in pixels
    max_cell_width: int = 1000  # Maximum cell width in pixels
    min_cell_height: int = 10  # Minimum cell height in pixels
    max_cell_height: int = 500  # Maximum cell height in pixels
    
    # Cell filtering - area constraints
    filter_by_area_ratio: bool = True  # Enable area ratio filtering
    min_area_ratio: float = 0.0001  # Minimum cell area / image area ratio
    max_area_ratio: float = 0.5  # Maximum cell area / image area ratio
    
    # Row and column assignment
    row_grouping_tolerance: float = 0.5  # Multiplier of mean height for row grouping
    column_center_tolerance: int = 4  # Divisor for column center distance calculation


class TableAnalysis:
    """
        Main class for extracting table cells from images using OpenCV.
    """

    def __init__(self, config: TableConfig = None):
        """Initialize TableAnalysis with configuration.
        
        Args:
            config: TableConfig instance. If None, uses default configuration.
        """
        if config is None:
            config = TableConfig()
        self.config = config.__dict__ if isinstance(config, TableConfig) else config

    def process(self, filepath: Path) -> List[BoxAnnotation]:
        """Process a single table image and extract cell positions.
        
        Args:
            filepath: Path to the input image file
            
        Returns:
            List of BoxAnnotation objects with cell positions and row/column indices
        """
        box_annotations: List[BoxAnnotation] = []

        # Step 1: Load and preprocess image
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {filepath}")
            return box_annotations
            
        img_height, img_width = img.shape
        img_area = img_width * img_height
        
        # Apply binary thresholding with Otsu's method
        thresh_val, img_bin = cv2.threshold(
            img, 
            self.config['binary_threshold'], 
            255, 
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        
        # Invert: black background, white lines/text
        img_bin = 255 - img_bin
        
        # Step 2: Define morphological kernels
        kernel_len = img_width // self.config['kernel_length_divisor']
        
        # Vertical kernel for detecting vertical lines
        ver_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.config['vertical_kernel_width'], kernel_len)
        )
        
        # Horizontal kernel for detecting horizontal lines
        hor_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (kernel_len, self.config['horizontal_kernel_height'])
        )
        
        # Small kernel for cleanup
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Step 3: Detect vertical and horizontal lines
        # Vertical lines
        img_temp = cv2.erode(img_bin, ver_kernel, iterations=self.config['erode_iterations'])
        vertical_lines = cv2.dilate(img_temp, ver_kernel, iterations=self.config['dilate_iterations'])
        
        # Horizontal lines
        img_temp = cv2.erode(img_bin, hor_kernel, iterations=self.config['erode_iterations'])
        horizontal_lines = cv2.dilate(img_temp, hor_kernel, iterations=self.config['dilate_iterations'])
        
        # Step 4: Combine lines to form table grid
        img_vh = cv2.addWeighted(
            vertical_lines, 
            self.config['vertical_line_weight'],
            horizontal_lines, 
            self.config['horizontal_line_weight'],
            0.0
        )
        
        # Erode and threshold the combined image
        img_vh = cv2.erode(~img_vh, cleanup_kernel, iterations=self.config['final_erode_iterations'])
        thresh_val, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Step 5: Find contours and extract cell bounding boxes
        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract and filter bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by width
            if w < self.config['min_cell_width'] or w > self.config['max_cell_width']:
                continue
            
            # Filter by height
            if h < self.config['min_cell_height'] or h > self.config['max_cell_height']:
                continue
            
            # Filter by area ratio (optional)
            if self.config['filter_by_area_ratio']:
                area_ratio = (w * h) / img_area
                if area_ratio < self.config['min_area_ratio'] or area_ratio > self.config['max_area_ratio']:
                    continue
            
            boxes.append([x, y, w, h])
        
        if len(boxes) == 0:
            print(f"Warning: No valid cells found in {filepath}")
            return box_annotations
        
        # Step 6: Group boxes into rows
        rows = self._group_into_rows(boxes)
        
        # Step 7: Assign column indices
        row_col_assignments = self._assign_columns(rows)
        
        # Step 8: Create BoxAnnotations
        for row_idx, row_data in enumerate(row_col_assignments):
            for col_idx, cells in row_data.items():
                for cell in cells:
                    x, y, w, h = cell
                    box_annotations.append(
                        BoxAnnotation(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            class_name=f"cell-{row_idx}-{col_idx}"
                        )
                    )
        
        return box_annotations

    def _group_into_rows(self, boxes: List[List[int]]) -> List[List[List[int]]]:
        """Group boxes into rows based on y-coordinate proximity.
        
        Args:
            boxes: List of [x, y, w, h] bounding boxes
            
        Returns:
            List of rows, where each row is a list of boxes sorted by x-coordinate
        """
        if len(boxes) == 0:
            return []
        
        # Sort boxes by y-coordinate (top to bottom)
        boxes = sorted(boxes, key=lambda b: b[1])
        
        # Calculate mean height for row grouping
        heights = [box[3] for box in boxes]
        mean_height = np.mean(heights)
        tolerance = mean_height * self.config['row_grouping_tolerance']
        
        # Group boxes into rows
        rows = []
        current_row = [boxes[0]]
        
        for i in range(1, len(boxes)):
            # Check if this box is in the same row as the previous one
            if abs(boxes[i][1] - current_row[-1][1]) <= tolerance:
                current_row.append(boxes[i])
            else:
                # Start a new row
                rows.append(sorted(current_row, key=lambda b: b[0]))  # Sort by x
                current_row = [boxes[i]]
        
        # Add the last row
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
        
        return rows

    def _assign_columns(self, rows: List[List[List[int]]]) -> List[dict]:
        """Assign column indices to boxes using a reference row approach.
        
        This handles merged cells and incomplete rows by using the row
        with the most cells as a reference for column positions.
        
        Args:
            rows: List of rows, where each row is a list of [x, y, w, h] boxes
            
        Returns:
            List of dictionaries mapping column_idx -> list of boxes for each row
        """
        if len(rows) == 0:
            return []
        
        # Find the row with the most cells (reference row)
        reference_row_idx = max(range(len(rows)), key=lambda i: len(rows[i]))
        reference_row = rows[reference_row_idx]
        
        # Calculate column centers from reference row
        column_centers = []
        for box in reference_row:
            x, y, w, h = box
            center_x = x + w // self.config['column_center_tolerance']
            column_centers.append(center_x)
        
        column_centers = sorted(column_centers)
        
        # Assign boxes to columns in each row
        result = []
        for row in rows:
            row_dict = {}
            
            for box in row:
                x, y, w, h = box
                box_center = x + w // self.config['column_center_tolerance']
                
                # Find nearest column center
                distances = [abs(box_center - center) for center in column_centers]
                col_idx = distances.index(min(distances))
                
                if col_idx not in row_dict:
                    row_dict[col_idx] = []
                row_dict[col_idx].append(box)
            
            result.append(row_dict)
        
        return result

    def write_results(self, box_annotations: List[BoxAnnotation], filepath: Path, output_dir: Path):
        """Write extracted cells as individual image files.
        
        Args:
            box_annotations: List of cell annotations to save
            filepath: Path to the original image file
            output_dir: Directory to save the extracted cell images
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load original color image
        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not read image {filepath} for cell extraction")
            return
            
        image_height, image_width = image.shape[:2]

        # Extract and save each cell
        for annotation in box_annotations:
            try:
                # Clamp coordinates to image boundaries
                x1 = max(0, annotation.x)
                y1 = max(0, annotation.y)
                x2 = min(image_width, x1 + annotation.width)
                y2 = min(image_height, y1 + annotation.height)

                # Skip invalid regions
                if x2 <= x1 or y2 <= y1:
                    continue

                # Extract cell region
                cell = image[y1:y2, x1:x2]

                # Save cell image
                output_path = output_dir / f"{annotation.class_name}.png"
                cv2.imwrite(str(output_path), cell)
                
            except Exception as e:
                print(f"Warning: Failed to extract cell {annotation.class_name}: {e}")
                continue


def main(table_dir: str, result_dir: str, config: TableConfig = None):
    """Process all table images in a directory.
    
    Args:
        table_dir: Directory containing input table images (.png, .jpg)
        result_dir: Directory to save extracted cells
        config: Optional TableConfig for custom parameters
        
    """
    # Initialize table analysis with config
    table_analysis = TableAnalysis(config)

    # Find all image files (PNG and JPG)
    table_dir_path = Path(table_dir)
    tables = sorted(list(table_dir_path.glob("*.png")) + list(table_dir_path.glob("*.jpg")))
    
    if len(tables) == 0:
        print(f"No table images found in {table_dir}")
        return

    # Process each table image
    for filepath in tqdm(tables, desc="Processing Tables", unit="tables"):
        box_annotations = table_analysis.process(filepath)
        output_dir = Path(result_dir) / filepath.stem
        table_analysis.write_results(box_annotations, filepath, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract table cells from images using OpenCV",
        epilog="""
Examples:
  # Use default settings
  python extract_cells.py tables results
  
  # Specify custom directories
  python extract_cells.py /path/to/tables /path/to/results
  
Configuration:
  To customize extraction parameters, modify the code or use the TableConfig class:
  
  from extract_cells import TableConfig, main
  config = TableConfig(row_grouping_tolerance=0.3, kernel_length_divisor=80)
  main("tables", "results", config)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "table_dir", 
        type=str, 
        nargs="?", 
        default="./tables",
        help="Directory containing input table images (default: ./tables)"
    )
    parser.add_argument(
        "result_dir", 
        type=str, 
        nargs="?", 
        default="./results",
        help="Directory to save extracted cells (default: ./results)"
    )
    args = parser.parse_args()
    main(table_dir=args.table_dir, result_dir=args.result_dir)
