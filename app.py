"""
Interactive App for Table Cell Extraction -- Streamlit-based interface to upload table images, 
configure extraction parameters, visualize results, and export extracted cells and metrics. 
Supports single image processing with live feedback and batch processing for multiple images 
with comprehensive reporting. Designed for ease of use and flexibility in handling various table formats.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import zipfile
import io
import csv
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import pandas as pd
from collections import defaultdict
import os

from extract_cells import TableAnalysis, TableConfig, BoxAnnotation
from validate_extraction import analyze_results

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Table Cell Extraction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Interactive Table Cell Extraction")
st.markdown("Extract table cells from images with customizable parameters and live visualization")

# ============================================================================
# SIDEBAR: CONFIGURATION PANEL
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Presets
    st.subheader("Quick Presets")
    preset = st.radio(
        "Select a preset configuration:",
        ["Default", "Dense Forms", "Sparse Tables", "Custom"],
        help="Quick-start configurations for different table types"
    )
    
    # Load preset or create custom config
    if preset == "Default":
        config_dict = {
            'kernel_length_divisor': 100,
            'row_grouping_tolerance': 0.5,
            'column_center_tolerance': 4,
            'binary_threshold': 128,
            'min_cell_width': 10,
            'min_cell_height': 10,
            'max_cell_width': 1000,
            'max_cell_height': 500,
            'erode_iterations': 3,
            'dilate_iterations': 3,
        }
    elif preset == "Dense Forms":
        config_dict = {
            'kernel_length_divisor': 80,
            'row_grouping_tolerance': 0.3,
            'column_center_tolerance': 3,
            'binary_threshold': 128,
            'min_cell_width': 15,
            'min_cell_height': 15,
            'max_cell_width': 800,
            'max_cell_height': 400,
            'erode_iterations': 4,
            'dilate_iterations': 4,
        }
    elif preset == "Sparse Tables":
        config_dict = {
            'kernel_length_divisor': 120,
            'row_grouping_tolerance': 0.7,
            'column_center_tolerance': 6,
            'binary_threshold': 128,
            'min_cell_width': 20,
            'min_cell_height': 20,
            'max_cell_width': 1500,
            'max_cell_height': 800,
            'erode_iterations': 3,
            'dilate_iterations': 3,
        }
    else:  # Custom
        config_dict = {}
    
    st.divider()
    st.subheader("📐 Fine-tune Parameters")
    
    # Image Processing
    with st.expander("Image Processing", expanded=True):
        config_dict['binary_threshold'] = st.slider(
            "Binary Threshold",
            min_value=0,
            max_value=255,
            value=config_dict.get('binary_threshold', 128),
            help="Threshold for binary conversion (0-255)"
        )
    
    # Line Detection
    with st.expander("Line Detection", expanded=True):
        config_dict['kernel_length_divisor'] = st.slider(
            "Kernel Length Divisor",
            min_value=50,
            max_value=200,
            value=config_dict.get('kernel_length_divisor', 100),
            step=5,
            help="Image width divided by this value = kernel length. Lower = finer lines"
        )
        
        config_dict['erode_iterations'] = st.slider(
            "Erode Iterations",
            min_value=1,
            max_value=10,
            value=config_dict.get('erode_iterations', 3),
            help="Number of erosion iterations for line detection"
        )
        
        config_dict['dilate_iterations'] = st.slider(
            "Dilate Iterations",
            min_value=1,
            max_value=10,
            value=config_dict.get('dilate_iterations', 3),
            help="Number of dilation iterations for line detection"
        )
    
    # Cell Filtering
    with st.expander("Cell Filtering", expanded=False):
        config_dict['min_cell_width'] = st.slider(
            "Min Cell Width (px)",
            min_value=5,
            max_value=50,
            value=config_dict.get('min_cell_width', 10),
            help="Minimum cell width in pixels"
        )
        
        config_dict['max_cell_width'] = st.slider(
            "Max Cell Width (px)",
            min_value=500,
            max_value=2000,
            value=config_dict.get('max_cell_width', 1000),
            help="Maximum cell width in pixels"
        )
        
        config_dict['min_cell_height'] = st.slider(
            "Min Cell Height (px)",
            min_value=5,
            max_value=50,
            value=config_dict.get('min_cell_height', 10),
            help="Minimum cell height in pixels"
        )
        
        config_dict['max_cell_height'] = st.slider(
            "Max Cell Height (px)",
            min_value=200,
            max_value=1000,
            value=config_dict.get('max_cell_height', 500),
            help="Maximum cell height in pixels"
        )
    
    # Row and Column Assignment
    with st.expander("Row & Column Assignment", expanded=False):
        config_dict['row_grouping_tolerance'] = st.slider(
            "Row Grouping Tolerance",
            min_value=0.1,
            max_value=1.0,
            value=config_dict.get('row_grouping_tolerance', 0.5),
            step=0.1,
            help="Multiplier of mean height for row grouping. Lower = stricter"
        )
        
        config_dict['column_center_tolerance'] = st.slider(
            "Column Center Tolerance",
            min_value=1,
            max_value=10,
            value=config_dict.get('column_center_tolerance', 4),
            help="Divisor for column center distance. Lower = more strict"
        )

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Tab layout
tab_demo, tab_batch = st.tabs(["📷 Single Image", "📦 Batch Processing"])

with tab_demo:
    st.header("Single Image Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ Upload or Select Image")
        
        # Option to upload or use demo image
        use_demo = st.checkbox("Use demo image from tables/ folder", value=False)
        
        if use_demo:
            tables_dir = Path("tables")
            available_tables = sorted(list(tables_dir.glob("*.png")) + list(tables_dir.glob("*.jpg")))
            
            if available_tables:
                selected_table = st.selectbox(
                    "Select a table image:",
                    available_tables,
                    format_func=lambda x: x.name
                )
                image_path = selected_table
                st.success(f"✓ Selected: {image_path.name}")
            else:
                st.warning("No tables found in ./tables/ directory")
                image_path = None
        else:
            uploaded_file = st.file_uploader(
                "Upload a table image",
                type=["png", "jpg", "jpeg"],
                help="Upload a PNG or JPG image of a table"
            )
            
            if uploaded_file:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    image_path = Path(tmp.name)
                st.success(f"✓ Uploaded: {uploaded_file.name}")
            else:
                image_path = None
    
    with col2:
        st.subheader("2️⃣ Process Image")
        
        # Choose processing mode
        processing_mode = st.radio(
            "Select mode:",
            ["Single Extraction", "Compare Presets"],
            horizontal=True,
            help="Single: Extract with current config | Compare: Run all 3 presets side-by-side"
        )
        
        if image_path and image_path.exists():
            if processing_mode == "Single Extraction":
                if st.button("▶️ Run Extraction", key="extract_button", use_container_width=True):
                    st.session_state['extraction_done'] = True
                    st.session_state['image_path'] = image_path
                    st.session_state['config_dict'] = config_dict
                    st.session_state['comparison_done'] = False
            else:  # Compare Presets mode
                if st.button("🔄 Compare All Presets", key="compare_button", use_container_width=True):
                    st.session_state['comparison_done'] = True
                    st.session_state['comparison_path'] = image_path
                    st.session_state['extraction_done'] = False
        else:
            st.info("📤 Upload or select an image first")
    
    # Display results if extraction was done
    if st.session_state.get('extraction_done') and st.session_state.get('image_path'):
        image_path = st.session_state['image_path']
        config_dict = st.session_state['config_dict']
        
        # Create TableConfig from dict
        config = TableConfig(**config_dict)
        analyzer = TableAnalysis(config)
        
        # Process the image
        with st.spinner("🔄 Extracting cells..."):
            box_annotations = analyzer.process(image_path)
        
        # Load original image
        original_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        st.divider()
        st.header("📊 Results")
        
        # Row 1: Original image with overlay + Metrics
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            # Draw boxes on image
            img_with_boxes = original_img_rgb.copy()
            
            for annotation in box_annotations:
                x1, y1 = annotation.x, annotation.y
                x2, y2 = annotation.x + annotation.width, annotation.y + annotation.height
                
                # Draw green rectangle
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            st.subheader("🎯 Input with Detection Overlay")
            st.image(img_with_boxes, use_container_width=True)
            
            # Download button for annotated image
            pil_img = Image.fromarray(img_with_boxes)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG')
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=img_bytes.getvalue(),
                file_name="annotated_table.png",
                mime="image/png",
                use_container_width=True
            )
        
        with result_col2:
            st.subheader("📈 Metrics")
            
            # Parse row/column info
            rows = defaultdict(list)
            cols = defaultdict(list)
            for ann in box_annotations:
                parts = ann.class_name.split('-')
                if len(parts) == 3:
                    row_idx = int(parts[1])
                    col_idx = int(parts[2])
                    rows[row_idx].append(col_idx)
                    cols[col_idx].append(row_idx)
            
            num_cells = len(box_annotations)
            num_rows = len(rows) if rows else 0
            num_cols = len(cols) if cols else 0
            expected_cells = num_rows * num_cols if num_rows > 0 and num_cols > 0 else 1
            coverage = (num_cells / expected_cells * 100) if expected_cells > 0 else 0
            
            # Status determination
            if coverage >= 95:
                status = "✅ EXCELLENT"
            elif coverage >= 80:
                status = "✨ GOOD"
            elif coverage >= 60:
                status = "⚠️ FAIR"
            else:
                status = "❌ POOR"
            
            st.metric("Total Cells", num_cells)
            st.metric("Grid Size", f"{num_rows} × {num_cols}" if num_rows > 0 else "0 × 0")
            st.metric("Coverage", f"{coverage:.1f}%")
            st.metric("Status", status)
        
        # Row 2: Cell Gallery
        st.divider()
        st.subheader("🖼️ Extracted Cells Gallery")
        
        # Extract cells
        image_height, image_width = original_img_rgb.shape[:2]
        extracted_cells = []
        
        for annotation in box_annotations:
            x1 = max(0, annotation.x)
            y1 = max(0, annotation.y)
            x2 = min(image_width, annotation.x + annotation.width)
            y2 = min(image_height, annotation.y + annotation.height)
            
            if x2 > x1 and y2 > y1:
                cell_img = original_img_rgb[y1:y2, x1:x2]
                extracted_cells.append({
                    'image': cell_img,
                    'label': annotation.class_name,
                    'row': int(annotation.class_name.split('-')[1]),
                    'col': int(annotation.class_name.split('-')[2]),
                    'width': annotation.width,
                    'height': annotation.height,
                })
        
        # Sort options
        sort_by = st.radio(
            "Sort cells by:",
            ["Row, then Column", "Column, then Row", "Size (largest first)"],
            horizontal=True
        )
        
        if sort_by == "Row, then Column":
            extracted_cells.sort(key=lambda x: (x['row'], x['col']))
        elif sort_by == "Column, then Row":
            extracted_cells.sort(key=lambda x: (x['col'], x['row']))
        else:
            extracted_cells.sort(key=lambda x: x['width'] * x['height'], reverse=True)
        
        # Display cells in grid
        cols_per_row = st.slider("Cells per row", min_value=2, max_value=10, value=5)
        
        for i in range(0, len(extracted_cells), cols_per_row):
            cols_grid = st.columns(cols_per_row)
            for j, col in enumerate(cols_grid):
                if i + j < len(extracted_cells):
                    cell = extracted_cells[i + j]
                    with col:
                        st.image(cell['image'], use_container_width=True)
                        st.caption(f"{cell['label']} ({cell['width']}×{cell['height']}px)")
        
        # Row 3: Export Options
        st.divider()
        st.subheader("💾 Export Results")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Export cells as ZIP
            if st.button("📦 Export Cells as ZIP", use_container_width=True):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for cell in extracted_cells:
                        cell_pil = Image.fromarray(cell['image'])
                        cell_bytes = io.BytesIO()
                        cell_pil.save(cell_bytes, format='PNG')
                        zip_file.writestr(f"{cell['label']}.png", cell_bytes.getvalue())
                
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="extracted_cells.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        with export_col2:
            # Export coordinates as CSV
            if st.button("📊 Export Coordinates as CSV", use_container_width=True):
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow(['Cell', 'Row', 'Col', 'X', 'Y', 'Width', 'Height'])
                
                for ann in sorted(box_annotations, key=lambda a: (
                    int(a.class_name.split('-')[1]), 
                    int(a.class_name.split('-')[2])
                )):
                    row = int(ann.class_name.split('-')[1])
                    col = int(ann.class_name.split('-')[2])
                    writer.writerow([ann.class_name, row, col, ann.x, ann.y, ann.width, ann.height])
                
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="cell_coordinates.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_col3:
            # Export summary as JSON
            if st.button("📋 Export Summary", use_container_width=True):
                summary = {
                    'image': image_path.name,
                    'total_cells': num_cells,
                    'grid_rows': num_rows,
                    'grid_cols': num_cols,
                    'coverage_percent': coverage,
                    'status': status,
                    'parameters': config_dict
                }
                
                st.download_button(
                    label="⬇️ Download Summary",
                    data=pd.DataFrame([summary]).to_json(orient='records'),
                    file_name="extraction_summary.json",
                    mime="application/json",
                    use_container_width=True
                )

    # ========================================================================
    # PRESET COMPARISON RESULTS
    # ========================================================================
    if st.session_state.get('comparison_done') and st.session_state.get('comparison_path'):
        comparison_path = st.session_state['comparison_path']
        
        # Load original image
        original_img = cv2.imread(str(comparison_path), cv2.IMREAD_COLOR)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        st.divider()
        st.header("🔄 Preset Comparison Results")
        
        # Define presets
        presets = {
            'Default': TableConfig(
                kernel_length_divisor=100,
                row_grouping_tolerance=0.5,
                column_center_tolerance=4,
                binary_threshold=128,
                min_cell_width=10,
                min_cell_height=10,
                max_cell_width=1000,
                max_cell_height=500,
                erode_iterations=3,
                dilate_iterations=3,
            ),
            'Dense Forms': TableConfig(
                kernel_length_divisor=80,
                row_grouping_tolerance=0.3,
                column_center_tolerance=3,
                binary_threshold=128,
                min_cell_width=15,
                min_cell_height=15,
                max_cell_width=800,
                max_cell_height=400,
                erode_iterations=4,
                dilate_iterations=4,
            ),
            'Sparse Tables': TableConfig(
                kernel_length_divisor=120,
                row_grouping_tolerance=0.7,
                column_center_tolerance=6,
                binary_threshold=128,
                min_cell_width=20,
                min_cell_height=20,
                max_cell_width=1500,
                max_cell_height=800,
                erode_iterations=3,
                dilate_iterations=3,
            ),
        }
        
        # Process with all presets
        st.info("🔄 Processing with all presets... This may take a few moments.")
        comparison_results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (preset_name, config) in enumerate(presets.items()):
            status_text.text(f"Processing {preset_name}...")
            analyzer = TableAnalysis(config)
            boxes = analyzer.process(comparison_path)
            
            # Calculate metrics
            rows = defaultdict(list)
            cols = defaultdict(list)
            for ann in boxes:
                parts = ann.class_name.split('-')
                if len(parts) == 3:
                    row_idx = int(parts[1])
                    col_idx = int(parts[2])
                    rows[row_idx].append(col_idx)
                    cols[col_idx].append(row_idx)
            
            num_cells = len(boxes)
            num_rows = len(rows) if rows else 0
            num_cols = len(cols) if cols else 0
            expected_cells = num_rows * num_cols if num_rows > 0 and num_cols > 0 else 1
            coverage = (num_cells / expected_cells * 100) if expected_cells > 0 else 0
            
            # Status determination
            if coverage >= 95:
                status = "✅ EXCELLENT"
            elif coverage >= 80:
                status = "✨ GOOD"
            elif coverage >= 60:
                status = "⚠️ FAIR"
            else:
                status = "❌ POOR"
            
            comparison_results[preset_name] = {
                'boxes': boxes,
                'num_cells': num_cells,
                'grid_rows': num_rows,
                'grid_cols': num_cols,
                'coverage': coverage,
                'status': status,
                'config': config
            }
            
            progress_bar.progress((idx + 1) / len(presets))
        
        status_text.text("✅ All presets processed!")
        progress_bar.empty()
        
        st.divider()
        st.subheader("📊 Side-by-Side Comparison")
        
        # Display side-by-side results
        cols = st.columns(3)
        
        for col_idx, (preset_name, result) in enumerate(comparison_results.items()):
            with cols[col_idx]:
                st.write(f"**{preset_name}**")
                
                # Draw boxes on image
                img_with_boxes = original_img_rgb.copy()
                for annotation in result['boxes']:
                    x1, y1 = annotation.x, annotation.y
                    x2, y2 = annotation.x + annotation.width, annotation.y + annotation.height
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                st.image(img_with_boxes, use_container_width=True)
                
                # Metrics
                st.metric("Cells", result['num_cells'])
                st.metric("Grid", f"{result['grid_rows']}×{result['grid_cols']}" if result['grid_rows'] > 0 else "0×0")
                st.metric("Coverage", f"{result['coverage']:.1f}%")
                st.metric("Status", result['status'])
        
        st.divider()
        st.subheader("📈 Detailed Comparison Table")
        
        # Create comparison dataframe
        comparison_data = []
        for preset_name, result in comparison_results.items():
            comparison_data.append({
                'Preset': preset_name,
                'Cells': result['num_cells'],
                'Grid': f"{result['grid_rows']}×{result['grid_cols']}" if result['grid_rows'] > 0 else "0×0",
                'Coverage %': f"{result['coverage']:.1f}",
                'Status': result['status'],
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Recommendation
        best_preset = max(comparison_results.items(), key=lambda x: x[1]['coverage'])
        st.divider()
        st.success(
            f"🏆 **Recommended Preset**: {best_preset[0]} "
            f"({best_preset[1]['coverage']:.1f}% coverage, {best_preset[1]['num_cells']} cells)"
        )
        
        # Option to use recommended preset
        if st.button(f"✨ Use '{best_preset[0]}' Preset", use_container_width=True):
            st.session_state['extraction_done'] = False
            st.session_state['comparison_done'] = False
            st.info(f"✓ {best_preset[0]} preset has been selected in the sidebar. Now click 'Run Extraction' to use it.")


with tab_batch:
    st.header("Batch Processing")
    
    st.info("📌 **How it works:** Upload a ZIP file containing multiple images or images from the tables/ folder. The app will extract cells from each image and provide a comprehensive report.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Batch")
        
        batch_option = st.radio("Processing mode:", ["Use tables/ folder", "Upload ZIP file"])
        
        if batch_option == "Use tables/ folder":
            tables_dir = Path("tables")
            available_tables = sorted(list(tables_dir.glob("*.png")) + list(tables_dir.glob("*.jpg")))
            
            if available_tables:
                st.write(f"Found **{len(available_tables)}** table images")
                
                # Allow selection
                all_selected = st.checkbox("Select all", value=True)
                if all_selected:
                    selected_files = available_tables
                else:
                    selected_files = st.multiselect(
                        "Select tables to process:",
                        available_tables,
                        format_func=lambda x: x.name,
                        default=available_tables[:5]
                    )
            else:
                st.warning("No tables found in ./tables/")
                selected_files = []
        else:
            zip_file = st.file_uploader("Upload ZIP file", type=['zip'])
            if zip_file:
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        with zipfile.ZipFile(zip_file) as z:
                            z.extractall(tmp_dir)
                        
                        tmp_path = Path(tmp_dir)
                        selected_files = list(tmp_path.glob("*.png")) + list(tmp_path.glob("*.jpg"))
                        st.success(f"✓ Extracted {len(selected_files)} images from ZIP")
                except Exception as e:
                    st.error(f"Error reading ZIP: {e}")
                    selected_files = []
            else:
                selected_files = []
    
    with col2:
        st.subheader("Process Batch")
        
        if selected_files:
            if st.button("▶️ Process All Images", use_container_width=True, key="batch_button"):
                st.session_state['batch_processing'] = True
                st.session_state['selected_files'] = selected_files
                st.session_state['batch_config'] = config_dict
    
    # Batch processing results
    if st.session_state.get('batch_processing'):
        selected_files = st.session_state['selected_files']
        config_dict = st.session_state['batch_config']
        
        config = TableConfig(**config_dict)
        analyzer = TableAnalysis(config)
        
        # Process all files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = {}
        
        for idx, file_path in enumerate(selected_files):
            status_text.text(f"Processing {idx + 1}/{len(selected_files)}: {file_path.name}")
            
            try:
                box_annotations = analyzer.process(file_path)
                all_results[file_path.name] = {
                    'path': file_path,
                    'annotations': box_annotations,
                    'error': None
                }
            except Exception as e:
                all_results[file_path.name] = {
                    'path': file_path,
                    'annotations': [],
                    'error': str(e)
                }
            
            progress_bar.progress((idx + 1) / len(selected_files))
        
        status_text.text("✅ Processing complete!")
        
        st.divider()
        st.header("📊 Batch Results Summary")
        
        # Create summary table
        summary_data = []
        for filename, result in all_results.items():
            annotations = result['annotations']
            if annotations:
                rows = defaultdict(list)
                cols = defaultdict(list)
                for ann in annotations:
                    parts = ann.class_name.split('-')
                    if len(parts) == 3:
                        row_idx = int(parts[1])
                        col_idx = int(parts[2])
                        rows[row_idx].append(col_idx)
                        cols[col_idx].append(row_idx)
                
                num_cells = len(annotations)
                num_rows = len(rows) if rows else 0
                num_cols = len(cols) if cols else 0
                expected = num_rows * num_cols
                coverage = (num_cells / expected * 100) if expected > 0 else 0
            else:
                num_cells = 0
                num_rows = 0
                num_cols = 0
                coverage = 0
            
            # Status
            if coverage >= 95:
                status = "✅ EXCELLENT"
            elif coverage >= 80:
                status = "✨ GOOD"
            elif coverage >= 60:
                status = "⚠️ FAIR"
            else:
                status = "❌ POOR"
            
            summary_data.append({
                'Image': filename,
                'Cells': num_cells,
                'Grid': f"{num_rows}×{num_cols}" if num_rows > 0 else "0×0",
                'Coverage %': f"{coverage:.1f}",
                'Status': status,
                'Error': result['error'] or '-'
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Export batch results
        st.divider()
        st.subheader("💾 Export Batch Results")
        
        export_batch_col1, export_batch_col2 = st.columns(2)
        
        with export_batch_col1:
            # Export all extracted cells
            if st.button("📦 Export all cells as ZIP", use_container_width=True):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, result in all_results.items():
                        if result['annotations']:
                            img = cv2.imread(str(result['path']), cv2.IMREAD_COLOR)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_height, img_width = img_rgb.shape[:2]
                            
                            for ann in result['annotations']:
                                x1 = max(0, ann.x)
                                y1 = max(0, ann.y)
                                x2 = min(img_width, ann.x + ann.width)
                                y2 = min(img_height, ann.y + ann.height)
                                
                                if x2 > x1 and y2 > y1:
                                    cell_img = img_rgb[y1:y2, x1:x2]
                                    cell_pil = Image.fromarray(cell_img)
                                    cell_bytes = io.BytesIO()
                                    cell_pil.save(cell_bytes, format='PNG')
                                    
                                    cell_filename = f"{Path(filename).stem}/{ann.class_name}.png"
                                    zip_file.writestr(cell_filename, cell_bytes.getvalue())
                
                st.download_button(
                    label="⬇️ Download all cells ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="all_extracted_cells.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        with export_batch_col2:
            # Export summary
            if st.button("📊 Export summary CSV", use_container_width=True):
                csv_buffer = io.StringIO()
                summary_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="⬇️ Download summary CSV",
                    data=csv_buffer.getvalue(),
                    file_name="batch_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Initialize session state
if 'extraction_done' not in st.session_state:
    st.session_state['extraction_done'] = False
if 'batch_processing' not in st.session_state:
    st.session_state['batch_processing'] = False
if 'comparison_done' not in st.session_state:
    st.session_state['comparison_done'] = False
if 'comparison_path' not in st.session_state:
    st.session_state['comparison_path'] = None
