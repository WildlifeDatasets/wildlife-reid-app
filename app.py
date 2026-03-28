import torch # Import torch first to avoid DLL issues
import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QListWidget, 
                             QLabel, QProgressBar, QGridLayout, QScrollArea, 
                             QLineEdit, QSplitter, QFrame, QMessageBox, QTabWidget)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
from engine import EmbeddingEngine
from data_model import DataManager

class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)
    def mousePressEvent(self, event):
        if self.objectName():
            self.clicked.emit(self.objectName())

class EmbeddingWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, engine, image_paths, bboxes=None):
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths
        self.bboxes = bboxes

    def run(self):
        embeddings = self.engine.generate_embeddings(self.image_paths, self.bboxes, self.progress.emit)
        self.finished.emit(embeddings)

class ValidationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Query image at the top
        self.query_info = QLabel("Query Image")
        self.layout.addWidget(self.query_info)
        self.query_image = QLabel()
        self.query_image.setFixedSize(300, 300)
        self.query_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.query_image)
        
        # ID input and update button
        self.id_layout = QHBoxLayout()
        self.id_label = QLabel("Current ID:")
        self.id_input = QLineEdit()
        self.update_btn = QPushButton("Update ID")
        self.id_layout.addWidget(self.id_label)
        self.id_layout.addWidget(self.id_input)
        self.id_layout.addWidget(self.update_btn)
        self.layout.addLayout(self.id_layout)
        
        # Matches grid
        self.matches_label = QLabel("Closest Matches")
        self.layout.addWidget(self.matches_label)
        
        self.scroll = QScrollArea()
        self.scroll_content = QWidget()
        self.matches_grid = QGridLayout(self.scroll_content)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll)

class SalamanderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Salamander Re-ID Tool")
        self.resize(1300, 900)
        
        self.apply_styles()
        
        self.data_manager = DataManager()
        self.engine = EmbeddingEngine()
        self.neighbor_indices = None
        self.neighbor_distances = None
        self.current_idx = None # Track current image being validated
        
        # Performance optimizations
        self.thumbnail_cache = {} # Cache for 160x160 pixmaps
        self.displayed_count = 50 # Initial count for pagination
        self.current_identities_list = [] # Store currently filtered IDs
        
        self.init_ui()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget#sidebar {
                background-color: #2c3e50;
                border-right: 1px solid #bdc3c7;
            }
            QWidget#sidebar QLabel {
                color: #ecf0f1;
                font-weight: bold;
                font-size: 13px;
                margin-top: 10px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
            QPushButton#back_btn {
                background-color: #e67e22;
            }
            QPushButton#back_btn:hover {
                background-color: #d35400;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #3498db;
            }
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #ecf0f1;
                border: 1px solid #bdc3c7;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                color: #2c3e50;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                width: 20px;
            }
            QFrame#card {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 8px;
            }
            QFrame#card:hover {
                border: 1px solid #3498db;
            }
            QLabel#card_id {
                font-weight: bold;
                color: #2c3e50;
            }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        sidebar_layout.setSpacing(10)
        
        title = QLabel("SALAMANDER RE-ID")
        title.setStyleSheet("font-size: 18px; color: #3498db; margin-bottom: 20px;")
        sidebar_layout.addWidget(title)
        
        self.load_btn = QPushButton("Load Metadata")
        self.load_btn.clicked.connect(self.load_data)
        sidebar_layout.addWidget(self.load_btn)
        
        self.back_btn = QPushButton("Back to Identities")
        self.back_btn.setObjectName("back_btn")
        self.back_btn.clicked.connect(lambda: self.refresh_grid())
        self.back_btn.setVisible(False)
        sidebar_layout.addWidget(self.back_btn)

        sidebar_layout.addSpacing(10)
        self.search_label = QLabel("SEARCH IDENTITY ID:")
        sidebar_layout.addWidget(self.search_label)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.filter_identities)
        sidebar_layout.addWidget(self.search_input)
        sidebar_layout.addSpacing(20)

        self.image_root_btn = QPushButton("Set Image Root")
        self.image_root_btn.clicked.connect(self.set_image_root)
        sidebar_layout.addWidget(self.image_root_btn)
        
        self.embed_btn = QPushButton("Generate Embeddings")
        self.embed_btn.clicked.connect(self.run_embeddings)
        self.embed_btn.setEnabled(False)
        sidebar_layout.addWidget(self.embed_btn)
        
        self.split_btn = QPushButton("Split Train/Test")
        self.split_btn.clicked.connect(self.run_split)
        self.split_btn.setEnabled(False)
        sidebar_layout.addWidget(self.split_btn)
        
        self.error_btn = QPushButton("Detect Errors")
        self.error_btn.clicked.connect(self.run_error_detection)
        self.error_btn.setEnabled(False)
        sidebar_layout.addWidget(self.error_btn)

        self.export_btn = QPushButton("Export Corrected CSV")
        self.export_btn.clicked.connect(self.export_data)
        self.export_btn.setEnabled(False)
        sidebar_layout.addWidget(self.export_btn)

        self.closest_set_btn = QPushButton("Run Closest Set Test")
        self.closest_set_btn.clicked.connect(self.run_closest_set_test)
        self.closest_set_btn.setEnabled(False)
        self.closest_set_btn.setStyleSheet("background-color: #9b59b6;") # Fialová barva pro odlišení
        sidebar_layout.addWidget(self.closest_set_btn)
        
        sidebar_layout.addStretch()
        
        self.status_label = QLabel("READY")
        self.status_label.setStyleSheet("color: #2ecc71; font-size: 11px; font-weight: bold; margin-bottom: 5px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.status_label)

        sidebar_layout.addWidget(QLabel("PROGRESS"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        sidebar_layout.addWidget(self.progress_bar)
        
        # Tabs for different views
        self.tabs = QTabWidget()
        self.tabs.setContentsMargins(10, 10, 10, 10)
        
        # Grid View
        self.grid_scroll = QScrollArea()
        self.grid_content = QWidget()
        self.grid_layout = QGridLayout(self.grid_content)
        self.grid_layout.setSpacing(15)
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setWidget(self.grid_content)
        self.tabs.addTab(self.grid_scroll, "Identities View")
        
        # Validation View
        self.validation_widget = ValidationWidget()
        self.validation_widget.update_btn.clicked.connect(self.update_current_id)
        self.tabs.addTab(self.validation_widget, "Validation Mode")
        
        # Error List View
        self.error_list = QListWidget()
        self.error_list.itemClicked.connect(self.load_error_case)
        self.tabs.addTab(self.error_list, "Potential Errors")
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.tabs)

        # Default paths
        self.image_root = "Salamanders_2025-04/images/SalamanderID2025/database/images"
        if not os.path.exists(self.image_root):
            self.image_root = os.path.join(os.getcwd(), self.image_root)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Metadata", "", "Metadata (*.csv *.xlsx)")
        if file_path:
            self.status_label.setText("LOADING METADATA...")
            self.status_label.setStyleSheet("color: #f1c40f; font-size: 11px; font-weight: bold; margin-bottom: 5px;")
            QApplication.processEvents()
            
            self.data_manager.load_metadata(file_path, self.image_root)
            self.unique_ids_cache = None # Clear cache on new data
            self.refresh_grid()
            
            self.embed_btn.setEnabled(True)
            self.split_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            self.status_label.setText(f"LOADED {len(self.data_manager.df)} IMAGES")
            self.status_label.setStyleSheet("color: #2ecc71; font-size: 11px; font-weight: bold; margin-bottom: 5px;")
            QMessageBox.information(self, "Success", f"Loaded {len(self.data_manager.df)} images.")

    def set_image_root(self):
        root = QFileDialog.getExistingDirectory(self, "Select Image Root Directory")
        if root:
            self.image_root = root
            if self.data_manager.df is not None:
                self.status_label.setText("UPDATING PATHS...")
                QApplication.processEvents()
                self.data_manager.load_metadata(self.data_manager.file_path, self.image_root)
                self.refresh_grid()

    def filter_identities(self):
        search_text = self.search_input.text().strip().lower()
        if not search_text:
            self.refresh_grid()
            return
        
        df = self.data_manager.df
        if df is None: return
        
        all_unique = df['Identity ID'].unique()
        filtered = [uid for uid in all_unique if search_text in str(uid).lower()]
        self.refresh_grid(filtered_ids=filtered)

    def refresh_grid(self, identity_id=None, filtered_ids=None):
        self.status_label.setText("REFRESHING GRID...")
        self.status_label.setStyleSheet("color: #f1c40f; font-size: 11px; font-weight: bold; margin-bottom: 5px;")
        QApplication.processEvents()

        # Clear layout
        for i in reversed(range(self.grid_layout.count())): 
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
            
        df = self.data_manager.df
        if df is None: 
            self.status_label.setText("READY")
            return

        if identity_id is None:
            # Show one representative per unique identity
            self.back_btn.setVisible(False)
            self.search_input.setVisible(True)
            self.search_label.setVisible(True)
            self.tabs.setTabText(0, "Identities View")
            
            if filtered_ids is not None:
                unique_ids = filtered_ids
            else:
                if not hasattr(self, 'unique_ids_cache') or self.unique_ids_cache is None:
                    self.unique_ids_cache = df['Identity ID'].unique()
                unique_ids = self.unique_ids_cache
            
            row, col = 0, 0
            for uid in unique_ids[:200]: 
                row_data = df[df['Identity ID'] == uid].iloc[0]
                self._add_image_to_grid(row_data, row, col, is_identity_link=True)
                col += 1
                if col > 4:
                    col = 0
                    row += 1
            self.status_label.setText(f"SHOWING {len(unique_ids[:200])} IDENTITIES")
        else:
            # Show all images for a specific identity
            self.back_btn.setVisible(True)
            self.search_input.setVisible(False)
            self.search_label.setVisible(False)
            self.tabs.setTabText(0, f"Identity: {identity_id}")
            subset = df[df['Identity ID'] == identity_id]
            
            row, col = 0, 0
            for _, row_data in subset.iterrows():
                self._add_image_to_grid(row_data, row, col, is_identity_link=False)
                col += 1
                if col > 4:
                    col = 0
                    row += 1
            self.status_label.setText(f"ID {identity_id}: {len(subset)} IMAGES")

        self.status_label.setStyleSheet("color: #2ecc71; font-size: 11px; font-weight: bold; margin-bottom: 5px;")

    def _add_image_to_grid(self, row_data, row, col, is_identity_link=False):
        container = QFrame()
        container.setObjectName("card")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        img_path = row_data['full_path']
        label = ClickableLabel()
        
        if is_identity_link:
            label.setObjectName(str(row_data['Identity ID']))
            label.clicked.connect(self.refresh_grid)
            label.setToolTip("Click to see all images of this identity")
        else:
            # Click on a specific image opens Validation Mode
            # We store the absolute index in the dataframe
            df = self.data_manager.df
            idx = df[df['filename'] == row_data['filename']].index[0]
            label.setObjectName(str(idx))
            label.clicked.connect(self.open_validation_from_grid)
            label.setToolTip("Click to validate/edit this image")
            
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            label.setPixmap(pixmap.scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            label.setText("Image not found")
        
        layout.addWidget(label)
        
        id_text = f"ID: {row_data['Identity ID']}"
        if is_identity_link:
            count = len(self.data_manager.df[self.data_manager.df['Identity ID'] == row_data['Identity ID']])
            id_text += f" ({count} imgs)"
        
        info_label = QLabel(id_text)
        info_label.setObjectName("card_id")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        self.grid_layout.addWidget(container, row, col)

    def open_validation_from_grid(self, idx_str):
        idx = int(idx_str)
        self.current_idx = idx
        self.show_validation_case(idx)
        self.tabs.setCurrentIndex(1) # Switch to Validation Mode tab

    def update_current_id(self):
        if self.current_idx is None:
            QMessageBox.warning(self, "Warning", "No image selected for update.")
            return
            
        new_id = self.validation_widget.id_input.text().strip()
        if not new_id:
            QMessageBox.warning(self, "Warning", "Identity ID cannot be empty.")
            return

        self.data_manager.update_id(self.current_idx, new_id)
        self.unique_ids_cache = None # Clear cache because identities have changed
        
        QMessageBox.information(self, "Updated", f"Identity ID updated to {new_id}")
        self.refresh_grid()

    def run_embeddings(self):
        self.status_label.setText("GENERATING EMBEDDINGS...")
        self.status_label.setStyleSheet("color: #f1c40f; font-size: 11px; font-weight: bold; margin-bottom: 5px;")
        
        image_paths = self.data_manager.df['full_path'].tolist()
        bboxes = self.data_manager.get_bboxes()
        
        self.worker = EmbeddingWorker(self.engine, image_paths, bboxes)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_embeddings_finished)
        self.embed_btn.setEnabled(False)
        self.worker.start()

    def on_embeddings_finished(self, embeddings):
        self.embed_btn.setEnabled(True)
        self.error_btn.setEnabled(True)
        self.closest_set_btn.setEnabled(True)
        self.status_label.setText("EMBEDDINGS READY")
        self.status_label.setStyleSheet("color: #2ecc71; font-size: 11px; font-weight: bold; margin-bottom: 5px;")
        QMessageBox.information(self, "Success", "Embeddings generated successfully.")


    def run_split(self):
        try:
            train_idx, test_idx = self.data_manager.split_data()
            QMessageBox.information(self, "Split Complete", 
                                    f"Train: {len(train_idx)} images\nTest: {len(test_idx)} images\nSplit based on Identity ID.")
        except Exception as e:
            QMessageBox.warning(self, "Split Error", f"Failed to split data: {str(e)}\n\n(Tip: You need at least 2 different identities for splitting)")


    def run_error_detection(self):
        D, I = self.engine.find_nearest_neighbors(k=5)
        self.neighbor_indices = I
        self.neighbor_distances = D
        errors = self.data_manager.detect_errors(D, I)
        
        self.error_list.clear()
        for idx in errors:
            item_text = f"Image {idx}: {self.data_manager.df.iloc[idx]['filename']} (ID: {self.data_manager.df.iloc[idx]['Identity ID']})"
            self.error_list.addItem(item_text)
        
        QMessageBox.information(self, "Detection Complete", f"Found {len(errors)} potential errors.")
        self.tabs.setCurrentIndex(2) # Switch to Error List tab

    def load_error_case(self, item):
        # Extract index from item text (e.g., "Image 123: ..." or "FAIL - Image 123: ...")
        text = item.text()
        try:
            if "Image " in text:
                parts = text.split("Image ")[1]
                idx_str = parts.split(":")[0].strip()
                # Remove any non-numeric characters just in case
                idx_str = "".join(filter(str.isdigit, idx_str))
                idx = int(idx_str)
                self.current_idx = idx
                self.show_validation_case(idx)
                self.tabs.setCurrentIndex(1) # Switch to Validation Mode tab
        except Exception as e:
            print(f"Error parsing index from list item: {e}")

    def show_validation_case(self, idx):
        row = self.data_manager.df.iloc[idx]
        self.validation_widget.query_info.setText(f"Query Image: {row['filename']}")
        
        # Load and scale query image with aspect ratio preserved
        pixmap = QPixmap(row['full_path'])
        if not pixmap.isNull():
            self.validation_widget.query_image.setPixmap(
                pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self.validation_widget.query_image.setText("Image not found")
            
        self.validation_widget.id_input.setText(str(row['Identity ID']))
        
        # Clear matches grid
        for i in reversed(range(self.validation_widget.matches_grid.count())):
            widget = self.validation_widget.matches_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
            
        # Show top neighbors
        if self.neighbor_indices is not None:
            # The indices are based on the full dataframe
            neighbors = self.neighbor_indices[idx]
            valid_neighbors = []
            for i, n_idx in enumerate(neighbors):
                if n_idx != idx:
                    valid_neighbors.append((n_idx, self.neighbor_distances[idx][i]))
            
            for i, (n_idx, dist) in enumerate(valid_neighbors[:4]):
                n_row = self.data_manager.df.iloc[n_idx]
                container = QWidget()
                layout = QVBoxLayout(container)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                img_label = QLabel()
                img_label.setFixedSize(200, 200)
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                n_pixmap = QPixmap(n_row['full_path'])
                if not n_pixmap.isNull():
                    img_label.setPixmap(n_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                else:
                    img_label.setText("N/A")
                layout.addWidget(img_label)
                
                info_label = QLabel(f"Match {i+1}\nID: {n_row['Identity ID']}\nDist: {dist:.4f}")
                info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(info_label)
                
                self.validation_widget.matches_grid.addWidget(container, 0, i)

    def export_data(self):
        try:
            path = "corrected_metadata.csv"
            
            # Pokud soubor existuje a nejde do něj psát (např. otevřen v Excelu), zkusíme jiný název
            if os.path.exists(path):
                try:
                    with open(path, 'a'): pass
                except PermissionError:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"corrected_metadata_{timestamp}.csv"
            
            self.data_manager.export_csv(path)
            abs_path = os.path.abspath(path)
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Data exported as:\n\n{path}\n\nFull path:\n{abs_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"An error occurred during export:\n{str(e)}"
            )

    def run_closest_set_test(self):
        result = self.data_manager.run_closest_set_validation(self.engine)
        if isinstance(result, str):
            QMessageBox.warning(self, "Chyba", result)
            return

        accuracy, results = result

        # Zobrazíme výsledky v seznamu chyb (Error List)
        self.error_list.clear()
        self.tabs.setTabText(2, "Test Results")

        for res in results:
            if not res["correct"]:
                q_id = self.data_manager.df.iloc[res['query_idx']]['Identity ID']
                m_id = self.data_manager.df.iloc[res['match_idx']]['Identity ID']
                # Format: Image 123: filename (ID: current_id) matched with Train ID: match_id
                item_text = f"FAIL - Image {res['query_idx']}: {self.data_manager.df.iloc[res['query_idx']]['filename']} (ID: {q_id}) -> Match ID: {m_id} (Score: {res['score']:.4f})"
                self.error_list.addItem(item_text)

        QMessageBox.information(self, "Test done",
                                f"Accurancy Closest Set: {accuracy:.2f}%\n"
                                f"Count of mistake is viewable in list.")
        self.tabs.setCurrentIndex(2)

    # def run_closest_set_test(self):
    #     result = self.data_manager.run_closest_set_validation(self.engine)
    #     if isinstance(result, str):
    #         QMessageBox.warning(self, "Chyba", result)
    #         return
    #
    #     accuracy, results = result
    #
    #     # Zobrazíme výsledky v seznamu chyb (Error List)
    #     self.error_list.clear()
    #     self.tabs.setTabText(2, "Test Results")
    #
    #     for res in results:
    #         if not res["correct"]:
    #             q_id = self.data_manager.df.iloc[res['query_idx']]['Identity ID']
    #             m_id = self.data_manager.df.iloc[res['match_idx']]['Identity ID']
    #             # Format: Image 123: filename (ID: current_id) matched with Train ID: match_id
    #             item_text = f"FAIL - Image {res['query_idx']}: {self.data_manager.df.iloc[res['query_idx']]['filename']} (ID: {q_id}) -> Match ID: {m_id} (Score: {res['score']:.4f})"
    #             self.error_list.addItem(item_text)
    #
    #     QMessageBox.information(self, "Test done",
    #                             f"Accurancy Closest Set: {accuracy:.2f}%\n"
    #                             f"Count of mistake is viewable in list.")
    #     self.tabs.setCurrentIndex(2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SalamanderApp()
    window.show()
    sys.exit(app.exec())
