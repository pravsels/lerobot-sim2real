#!/usr/bin/env python3
import sys
import os
import cv2
import json
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QTextEdit, QVBoxLayout, QHBoxLayout, QWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF
import argparse

# —— API Helpers ——

def load_api_client():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No GOOGLE_API_KEY in .env")
    return genai.Client(api_key=api_key)

# Prompt
SYSTEM_PROMPT = """
You are an expert at analyzing images for robotic pick-and-place tasks.
Detect and identify each object. Objects are ONLY either an orange cube or a container (bins, boxes).
Return a JSON array of objects, each formatted exactly as:
[
  {
    "label": "<object_name>",
    "box_2d": [ymin, xmin, ymax, xmax]
  }
]
where <object_name> is a descriptive identifier.
Coordinates must be integers normalized to the 0–1000 range.
Return ONLY the JSON array, without any markdown or additional text.
"""
MODEL_ID = "gemini-2.0-flash-lite-001"

# —— Utility functions ——

def cv2_to_pil(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def clean_json_response(raw_text):
    lines = raw_text.strip().splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if "```json" in line.lower(): start = i+1
        elif line.strip().startswith("```") and i>start:
            end = i; break
    return "\n".join(lines[start:end])


def analyze_frame(client, frame, prompt):
    pil = cv2_to_pil(frame)
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=[pil, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3
        )
    )
    text = clean_json_response(resp.text)
    return json.loads(text)

# —— GUI Components ——

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self.pixmap_item)
        self.rect_item = None
        self.origin = QPointF()
        self.drawing = False

    def load_image(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qt_img)
        self.pixmap_item.setPixmap(pix)
        self.setSceneRect(QRectF(0, 0, w, h))
        if self.rect_item:
            self._scene.removeItem(self.rect_item)
            self.rect_item = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.rect_item:
                self._scene.removeItem(self.rect_item)
                self.rect_item = None
            self.origin = self.mapToScene(event.pos())
            pen = QPen(QColor('red'), 2, Qt.DashLine)
            self.rect_item = self._scene.addRect(QRectF(self.origin, self.origin), pen)
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing and self.rect_item:
            rect = QRectF(self.origin, self.mapToScene(event.pos())).normalized()
            self.rect_item.setRect(rect)

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def get_roi(self):
        if self.rect_item:
            return self.rect_item.rect().toRect()
        return None

class MainWindow(QMainWindow):
    def __init__(self, frames_dir):
        super().__init__()
        self.client = load_api_client()
        self.folder = Path(frames_dir)
        self.frames = sorted(
            [p for p in self.folder.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg','.bmp')]
        )
        if not self.frames:
            raise RuntimeError("No image files in directory")
        self.idx = 0
        self.out_dir = self.folder.parent / 'frames_bbox'
        self.out_dir.mkdir(exist_ok=True)
        self.detections = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Frame Annotation')
        prev_btn = QPushButton('Prev')
        prev_btn.clicked.connect(self.prev_frame)
        next_btn = QPushButton('Next')
        next_btn.clicked.connect(self.next_frame)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFixedHeight(200)
        self.prompt_edit.setPlainText(SYSTEM_PROMPT)
        preview_btn = QPushButton('Preview Mask')
        preview_btn.clicked.connect(self.preview_mask)
        analyze_btn = QPushButton('Analyze')
        analyze_btn.clicked.connect(self.analyze)
        save_btn = QPushButton('Save')
        save_btn.clicked.connect(self.save)
        self.status = QLabel('')
        self.viewer = ImageViewer()
        btn_layout = QHBoxLayout()
        for w in (prev_btn, next_btn, preview_btn, analyze_btn, save_btn): btn_layout.addWidget(w)
        vbox = QVBoxLayout()
        vbox.addLayout(btn_layout)
        vbox.addWidget(QLabel('Prompt:'))
        vbox.addWidget(self.prompt_edit)
        vbox.addWidget(self.viewer)
        vbox.addWidget(self.status)
        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)
        self.showMaximized()
        self.load_frame()

    def load_frame(self):
        path = self.frames[self.idx]
        img = cv2.imread(str(path))
        self.current_img = img
        self.viewer.load_image(img)
        title = f"Frame Annotation — {self.idx+1}/{len(self.frames)}: {path.name}"
        self.setWindowTitle(title)
        self.status.setText('Ready')
        self.detections = []

    def prev_frame(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_frame()

    def next_frame(self):
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self.load_frame()

    def preview_mask(self):
        """Show exactly what gets sent to Gemini (masked outside the ROI)."""
        if not hasattr(self, 'current_img'):
            return

        roi = self.viewer.get_roi()
        if not roi:
            QMessageBox.information(self, "No ROI", "Draw a box first to preview masked image.")
            return

        x, y = roi.x(), roi.y()
        w, h = roi.width(), roi.height()

        # Make a copy and zero out everything outside the ROI
        masked = self.current_img.copy()
        black = masked.copy(); black[:] = 0
        masked[:, :] = black
        masked[int(y):int(y+h), int(x):int(x+w)] = \
            self.current_img[int(y):int(y+h), int(x):int(x+w)]

        # Display it
        self.viewer.load_image(masked)
        self.status.setText("Previewing masked image")

    def analyze(self):
        if hasattr(self, 'current_img'):
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.status.setText('Analyzing...')
            QApplication.processEvents()
            try:
                roi = self.viewer.get_roi()
                prompt = self.prompt_edit.toPlainText()
                if roi:
                    prompt += "\n\nIMPORTANT: ONLY detect objects inside the viewable region; ignore everything outside in the black part of the image. Its been blacked out for a reason, because the objects don't exist there!!! Please do not ignore this!!"
                    x, y = roi.x(), roi.y()
                    w, h = roi.width(), roi.height()
                    # mask out everything outside the ROI
                    img_masked = self.current_img.copy()
                    img_masked[:, :] = 0
                    img_masked[int(y):int(y+h), int(x):int(x+w)] = self.current_img[int(y):int(y+h), int(x):int(x+w)]
                    img_for_gemini = img_masked
                else:
                    img_for_gemini = self.current_img
                # update prompt display
                self.prompt_edit.setPlainText(prompt)
                # call Gemini
                self.detections = analyze_frame(self.client, img_for_gemini, prompt)
                # display results on original image
                self.viewer.load_image(self.current_img)
                display = self.current_img.copy()
                for obj in self.detections:
                    ymin, xmin, ymax, xmax = obj['box_2d']
                    H, W = display.shape[:2]
                    y1, x1 = int(ymin/1000 * H), int(xmin/1000 * W)
                    y2, x2 = int(ymax/1000 * H), int(xmax/1000 * W)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, obj.get('label', ''), (x1, max(y1-10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                self.viewer.load_image(display)
                self.status.setText(f'Analysis complete: {len(self.detections)} objects found')
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
                self.status.setText('Analysis failed')
            finally:
                QApplication.restoreOverrideCursor()

    def save(self):
        if not self.detections:
            return
        path = self.frames[self.idx]
        display = self.current_img.copy()
        for obj in self.detections:
            ymin, xmin, ymax, xmax = obj['box_2d']
            H, W = display.shape[:2]
            y1, x1 = int(ymin/1000 * H), int(xmin/1000 * W)
            y2, x2 = int(ymax/1000 * H), int(xmax/1000 * W)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, obj.get('label', ''), (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        img_dst = self.out_dir / path.name
        cv2.imwrite(str(img_dst), display)
        json_dst = self.out_dir / f"{path.stem}.json"
        with open(json_dst, 'w') as jf:
            json.dump(self.detections, jf, indent=2)
        self.status.setText(f'Saved image to {img_dst.name} and JSON to {json_dst.name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame annotation GUI')
    parser.add_argument('frames_dir', help='Path to frames directory')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.frames_dir)
    sys.exit(app.exec_())
