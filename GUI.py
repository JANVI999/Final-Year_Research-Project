import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QDateEdit, QSizePolicy
from PyQt5.QtCore import Qt, QDate
from datetime import timedelta
from modis_tools.auth import ModisSession
from modis_tools.resources import CollectionApi, GranuleApi
from modis_tools.granule_handler import GranuleHandler
from geopy.geocoders import Nominatim
from netCDF4 import Dataset
import numpy as np
import cv2
import keras.models
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

class ModisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.location_label = QLabel('Location:')
        self.location_edit = QLineEdit()
        self.start_date_label = QLabel('Start Date:')
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate())
        self.submit_button = QPushButton('Download MODIS Data')

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)  # Center-align the image

        layout = QVBoxLayout()
        layout.addWidget(self.location_label)
        layout.addWidget(self.location_edit)
        layout.addWidget(self.start_date_label)
        layout.addWidget(self.start_date_edit)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.image_label)  # Add image label to layout
        self.setLayout(layout)

        self.submit_button.clicked.connect(self.downloadModisData)

        self.setGeometry(100, 100, 1160, 640)
        self.setWindowTitle('MODIS Data Downloader')
        self.show()

    def load_single_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def update_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def predict_mask(self, model, image):
        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        binary_predictions = np.argmax(predictions, axis=-1)
        return np.squeeze(binary_predictions)

    def generate_image_from_nc(self, nc_file_path):
        with Dataset(nc_file_path, 'r') as nc_file:
            chlor_a_data = nc_file.groups['geophysical_data'].variables['chlor_a'][:]

            num_lines, pixels_per_line = chlor_a_data.shape

            plt.figure(figsize=(254/300, 255/300))

            plt.imshow(chlor_a_data, cmap='jet', origin='lower', aspect='auto',
                    extent=(0, pixels_per_line, 0, num_lines),
                    norm=SymLogNorm(linthresh=0.01, linscale=1, vmin=-100, vmax=100))

            plt.axis('off')
            plt.colorbar().remove()

            plt.savefig('chlorophyll_image.png', dpi=300, bbox_inches='tight')

            image_path = 'chlorophyll_image.png'

            self.update_image(image_path)  # Update the image label with the new image

    def get_bounding_box(self, location_name, margin=0.0):
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode(location_name)

        if location:
            min_lon = location.longitude - margin
            min_lat = location.latitude - margin
            max_lon = location.longitude + margin
            max_lat = location.latitude + margin
            return [min_lon, min_lat, max_lon, max_lat]
        else:
            return None

    def downloadModisData(self):
        location_name = self.location_edit.text()
        start_date = self.start_date_edit.date()

        # Parse the start date and calculate end date
        end_date = start_date.addDays(1)

        # Geocode the location
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode(location_name)

        if location:
            bounding_box = self.get_bounding_box(location_name)
            print("Bounding Box: {}".format(bounding_box))

            # MODIS authentication
            username = "Paridhi09"
            password = "ParidhiSB@09"
            session = ModisSession(username=username, password=password)

            collection_client = CollectionApi(session=session)
            collections = collection_client.query(short_name="MODISA_L2_OC", version="R2022.0")
            granule_client = GranuleApi.from_collection(collections[0], session=session)

            granules = granule_client.query(start_date=start_date.toString("yyyy-MM-dd"), end_date=end_date.toString("yyyy-MM-dd"),
                                        bounding_box=bounding_box)

            downloaded_files = GranuleHandler.download_from_granules(granules, session)
            nc_file_path = downloaded_files[0]

            self.generate_image_from_nc(nc_file_path)

        else:
            print("Location not found.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    modisApp = ModisApp()
    sys.exit(app.exec_())
