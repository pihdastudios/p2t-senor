import datetime
import uuid
from typing import Dict

import yaml
from PyQt6 import QtCore, QtGui, QtWidgets
from msrest.authentication import CognitiveServicesCredentials
from msrest.exceptions import ValidationError

from CosmosDbApi import CosmosDbApi
from FaceApi import *
from OpencvQt import Capture, Converter

config = {
    "DURATION_INT": 5
}
person_group_name = "mhs"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.capture = Capture()
        self.converter = Converter()
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QVBoxLayout(central_widget)
        self.view = QtWidgets.QLabel()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_send = QtWidgets.QPushButton("Scan")
        self.btn_train = QtWidgets.QPushButton("Train")
        self.course_txt = QtWidgets.QTextEdit("Advanced Engineering Mathematics")
        self.label_time = QtWidgets.QLabel()
        self.course_lbl = QtWidgets.QLabel()
        self.course_lbl.setText("Course")
        lay.addWidget(self.view, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_stop)
        lay.addWidget(self.btn_send)
        lay.addWidget(self.btn_train)
        lay.addWidget(self.course_lbl)
        lay.addWidget(self.course_txt)
        lay.addWidget(self.label_time, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.view.setFixedSize(640, 400)
        self.show()
        self.init_camera()
        self.init_email()

        # detect face

        self.face_client = prepare_face_api()
        self.cosmos_client = prepare_cosmosdb_api()

    def init_camera(self):
        capture_thread = QtCore.QThread(self)
        converter_thread = QtCore.QThread(self)
        self.converter.setProcessAll(False)
        capture_thread.start()
        converter_thread.start()
        self.capture.moveToThread(capture_thread)
        self.converter.moveToThread(converter_thread)
        self.capture.frameReady.connect(self.converter.processFrame)
        self.converter.imageReady.connect(self.setImage)
        self.capture.started.connect(lambda: print("started"))
        self.btn_train.clicked.connect(self.train)
        self.btn_start.clicked.connect(self.capture.start)
        self.btn_stop.clicked.connect(self.capture.stop)

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.view.setPixmap(QtGui.QPixmap.fromImage(image))

    def init_email(self):
        timeline = QtCore.QTimeLine(config["DURATION_INT"] * 1000, self)
        timeline.frameChanged.connect(self.on_frame_changed)
        timeline.setFrameRange(0, config["DURATION_INT"])
        timeline.setDirection(QtCore.QTimeLine.Direction.Backward)
        self.btn_send.clicked.connect(timeline.start)

    def on_frame_changed(self, frame):
        if frame != 0:
            self.label_time.setNum(frame)
        else:
            self.label_time.setText("Smile...!")
            QtWidgets.QApplication.beep()
            image = QtGui.QImage(self.converter.image)
            ba = QtCore.QByteArray()
            buff = QtCore.QBuffer(ba)
            image.save("./img/captured/cap.png", "PNG")

            # done training

            # # captured face
            cap_face = extract_faces_from_image(self.face_client, "./img/captured/cap.png")
            results = resolve_face_ids(self.face_client, cap_face[0], person_group_name)
            for person in results:
                # for everybody found, copy the file in a folder with his/her name
                if len(person.candidates) > 0:
                    most_likely_candidate = person.candidates[0]

                    confidence = most_likely_candidate.confidence
                    recognized_person = self.face_client.person_group_person.get(person_group_name,
                                                                                 most_likely_candidate.person_id)
                    recognized_name = recognized_person.name
                    print(f"Recognized {recognized_name} in image")

                    # save to cosmosdb
                    self.cosmos_client.add_entry(str(uuid.uuid4()), recognized_person.name, self.course_txt.toPlainText(), datetime.datetime.now())

    def closeEvent(self, event):
        self.capture.stop()
        super(MainWindow, self).closeEvent(event)

    def train(self):
        # from file storage
        # Create the person group if the credentials are valid and the naming is allowed
        # delete first
        # groups = self.face_client.person_group.list()
        # for group in groups:
        #     print(group)
        #     try:
        #         run_if_valid_credentials(self.face_client.person_group.delete, group['person_group_id'])
        #     except:
        #         pass
        # run_if_valid_credentials(self.face_client.person_group.create, person_group_name, person_group_name)

        add_images_to_person_group(self.face_client, person_group_name, "./img/reference/",
                                   "./img/resized/")
        train_person_group(self.face_client, person_group_name)


def prepare_face_api() -> FaceClient:
    yaml_path = sys.path[0] + "/config.yaml"
    secrets_dict = extract_infos_from_yaml(yaml_path, "secrets")
    key = secrets_dict["face_api_key"]
    endpoint = secrets_dict["face_api_endpoint"]
    try:
        return FaceClient(endpoint, CognitiveServicesCredentials(key))
    except ValueError:
        # click.echo("You need to provide an endpoint and API key in the config.yaml.")
        sys.exit()


def prepare_cosmosdb_api() -> CosmosDbApi:
    yaml_path = sys.path[0] + "/config.yaml"
    secrets_dict = extract_infos_from_yaml(yaml_path, "secrets")
    key = secrets_dict["cosmosdb_api_key"]
    endpoint = secrets_dict["cosmosdb_api_endpoint"]
    try:
        return CosmosDbApi(endpoint, key)
    except ValueError:
        # click.echo("You need to provide an endpoint and API key in the config.yaml.")
        sys.exit()


def extract_infos_from_yaml(yaml_path: str, yaml_part: str) -> Optional[Dict]:
    """Reads a yaml at the specified path and returns the part of the file that is specified as a dict
    Args:
        yaml_path (str): location of the file
        yaml_part (str): Part of the yaml (at the highest level) that should be returned
    Returns:
        Dict: [description]
    """
    yaml_lines = open(yaml_path).readlines()
    yaml_string = "".join(yaml_lines)
    yaml_dict = yaml.safe_load(yaml_string)
    try:
        return yaml_dict[yaml_part]
    except KeyError:
        print(f"The specified part {yaml_part} could not be found in file {yaml_path}.")


def run_if_valid_credentials(func, *args):
    try:
        func(*args)

    except APIErrorException:
        print("""API error. Either this person group already exists, or your credentials are invalid. Please try another name. If this error appears again,
                         please check your Microsoft Azure Face API endpoint and key in the config.yaml""")
        sys.exit()

    except ValidationError:
        print(
            "The person group name is invalid, please only use letters (no special characters like 'ä'), numbers and underscores")
        sys.exit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec())
