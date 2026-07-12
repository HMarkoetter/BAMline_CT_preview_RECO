import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
import tomopy
import math
import os
import threading
import cv2                                      #to install package with pycharm search for "opencv-python"
import pvaccess as pva                          #to install package with pycharm search for "pvapy"
import epics
import csv
import time



# On-the-fly Navigator
version =  "Version 2026.07.12"

os.environ["EPICS_CA_ADDR_LIST"] = "172.31.20.131 172.31.20.231 172.31.20.145"

#Install ImageJ-PlugIn: EPICS AreaDetector NTNDA-Viewer, look for the channel specified here under channel_name, consider multiple users on servers!!!
channel_name = 'BAMline:Navigator'
channel_name_rec = 'BAMline:NavigatorReco'
channel_name_proj = 'BAMline:NavigatorProj'

ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'on_the_fly_navigator.ui')
Ui_on_the_fly_Navigator_Window, Q_on_the_fly_Navigator_Window = loadUiType(ui_path)

class AngularStatusWidget(QtWidgets.QWidget):
    projection_received = QtCore.pyqtSignal(float, bool)
    history_reset = QtCore.pyqtSignal()
    stable_complete = QtCore.pyqtSignal()
    selection_changed = QtCore.pyqtSignal(int)

    def __init__(self, sector_count, projection_count, parent=None):
        super().__init__(parent)
        self.arc_count = min(72, max(12, int(sector_count)))
        self.projection_count = max(1, int(projection_count))
        self.arcs = numpy.zeros(self.arc_count, dtype=numpy.uint8)
        self.current_angle = 0.0
        self.selected_projection = 0
        self.selection_drag_active = False
        self.complete = False
        self.paint_pending = True
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.projection_received.connect(self._record_projection)
        self.history_reset.connect(self._reset_history)
        self.stable_complete.connect(self._set_complete)

        self.repaint_timer = QtCore.QTimer(self)
        self.repaint_timer.timeout.connect(self._repaint_if_pending)
        self.repaint_timer.start(33)

    def record_projection(self, angle, moving):
        self.projection_received.emit(float(angle), bool(moving))

    def reset_history(self):
        self.history_reset.emit()

    def mark_stable_complete(self):
        self.stable_complete.emit()

    @QtCore.pyqtSlot(int, int)
    def set_counts(self, sector_count, projection_count):
        self.arc_count = min(72, max(12, int(sector_count)))
        self.projection_count = max(1, int(projection_count))
        self.arcs = numpy.zeros(self.arc_count, dtype=numpy.uint8)
        self.selected_projection %= self.projection_count
        self.complete = False
        self.paint_pending = True

    @QtCore.pyqtSlot(float, bool)
    def _record_projection(self, angle, moving):
        self.current_angle = angle % 360.0
        position = int(self.current_angle / 360.0 * self.arc_count) % self.arc_count
        opposite = (position + self.arc_count // 2) % self.arc_count
        color_index = 1 if moving else 2
        self.arcs[position] = color_index
        self.arcs[opposite] = color_index
        self.paint_pending = True

    @QtCore.pyqtSlot()
    def _reset_history(self):
        self.arcs.fill(0)
        self.complete = False
        self.paint_pending = True

    @QtCore.pyqtSlot()
    def _set_complete(self):
        self.arcs.fill(2)
        self.complete = True
        self.paint_pending = True

    @QtCore.pyqtSlot()
    def _repaint_if_pending(self):
        if self.paint_pending:
            self.paint_pending = False
            self.update()

    def _draw_arc_runs(self, painter, circle, state, color, width):
        span = 360.0 / self.arc_count
        position = 0
        painter.setPen(QtGui.QPen(color, width, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))

        while position < self.arc_count:
            if self.arcs[position] != state:
                position += 1
                continue

            start = position
            while position < self.arc_count and self.arcs[position] == state:
                position += 1

            painter.drawArc(
                circle,
                -round((90.0 - start * span) * 16),
                -round(-(position - start) * span * 16),
            )

    def _center_geometry(self):
        side = min(self.width(), self.height()) - 8
        center = QtCore.QPointF(self.width() / 2, self.height() / 2)
        return side, center, side * 0.22

    def _selection_from_position(self, position, require_inside=True):
        side , center, center_radius = self._center_geometry()
        delta_x = position.x() - center.x()
        delta_y = position.y() - center.y()
        if require_inside and math.hypot(delta_x, delta_y) > side:
            return False

        angle = (90.0 - math.degrees(math.atan2(delta_y, delta_x))) % 360.0
        selected = int(round(angle / 360.0 * self.projection_count)) % self.projection_count
        if selected != self.selected_projection:
            self.selected_projection = selected
            self.paint_pending = True
            self.setToolTip(
                'Projection {}/{} ({:.1f} deg)'.format(
                    selected + 1,
                    self.projection_count,
                    selected * 360.0 / self.projection_count,
                )
            )
        return True

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._selection_from_position(event.localPos()):
            self.selection_drag_active = True
            self.setFocus()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selection_drag_active and event.buttons() & QtCore.Qt.LeftButton:
            self._selection_from_position(event.localPos(), require_inside=False)
            self.selection_changed.emit(self.selected_projection)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.selection_drag_active:
            self._selection_from_position(event.localPos(), require_inside=False)
            self.selection_drag_active = False
            self.selection_changed.emit(self.selected_projection)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        step = 1 if event.angleDelta().y() > 0 else -1
        self.selected_projection = (self.selected_projection + step) % self.projection_count
        self.paint_pending = True
        self.selection_changed.emit(self.selected_projection)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Down):
            self.selected_projection = (self.selected_projection - 1) % self.projection_count
        elif event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Up):
            self.selected_projection = (self.selected_projection + 1) % self.projection_count
        else:
            super().keyPressEvent(event)
            return

        self.paint_pending = True
        self.selection_changed.emit(self.selected_projection)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        side, center, center_radius = self._center_geometry()
        ring_width = max(12.0, side * 0.27)
        ring_diameter = side - ring_width
        circle = QtCore.QRectF(
            (self.width() - ring_diameter) / 2,
            (self.height() - ring_diameter) / 2,
            ring_diameter,
            ring_diameter,
        )

        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor('#d9dde3'), ring_width))
        painter.drawEllipse(circle)
        self._draw_arc_runs(painter, circle, 1, QtGui.QColor('#9c2aa0'), ring_width)
        self._draw_arc_runs(painter, circle, 2, QtGui.QColor('#49a72f'), ring_width)

        painter.setBrush(QtGui.QColor('#49a72f') if self.complete else QtGui.QColor('#e53935'))
        painter.setPen(QtGui.QPen(QtGui.QColor('#4a4a4a'), 1))
        painter.drawEllipse(center, center_radius, center_radius)

        angle = -math.radians(self.current_angle - 90.0)
        angle2 = -math.radians(self.current_angle - 90.0- 180)
        line_radius = side * 0.47
        endpoint = QtCore.QPointF(
            center.x() + math.cos(angle) * line_radius,
            center.y() + math.sin(angle) * line_radius,
        )
        endpoint2 = QtCore.QPointF(
            center.x() + math.cos(angle2) * line_radius,
            center.y() + math.sin(angle2) * line_radius,
        )
        painter.setPen(QtGui.QPen(QtGui.QColor('#202124'), 2))
        painter.drawLine(center, endpoint)
        painter.drawLine(center, endpoint2)

        selection_angle = -math.radians(
            self.selected_projection * 360.0 / self.projection_count - 90.0
        )
        #selection_radius = center_radius * 0.78
        selection_radius = side/2
        selection_endpoint = QtCore.QPointF(
            center.x() + math.cos(selection_angle) * selection_radius,
            center.y() + math.sin(selection_angle) * selection_radius,
        )
        painter.setPen(QtGui.QPen(QtGui.QColor('#202124'), 5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawLine(center, selection_endpoint)
        painter.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawLine(center, selection_endpoint)
        painter.setPen(QtGui.QPen(QtGui.QColor('#202124'), 1))
        painter.setBrush(QtGui.QColor('#ffffff'))
        painter.drawEllipse(selection_endpoint, 4, 4)

class OnTheFlyNavigator(Ui_on_the_fly_Navigator_Window, Q_on_the_fly_Navigator_Window):
    rotation_state_received = QtCore.pyqtSignal(object)
    distance_received = QtCore.pyqtSignal(object)
    acquisition_config_changed = QtCore.pyqtSignal(int, int, int, int)
    refresh_ui_requested = QtCore.pyqtSignal()


    def __init__(self):
        super(OnTheFlyNavigator, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('On-the-fly Navigator')
        self.rotation_state_received.connect(self._apply_rotation_state)
        self.distance_received.connect(self._apply_distance)
        self.acquisition_config_changed.connect(self._apply_acquisition_config_ui)
        self.refresh_ui_requested.connect(self.read_parameter)
        self.start_rotate.setText('Checking rotation...')
        self.start_rotate.setEnabled(False)
        self.rotation_running = False


        # create pva type pv for reconstruction by copying metadata from the data pv, but replacing the sizes
        # This way the ADViewer (NDViewer) plugin can be also used for visualizing reconstructions.
        #pva_image_data = self.pva_structure.get('')
        #pva_image_dict = pva_image_data.getStructureDict()
        pva_image_dict = {'value': ({'booleanValue': [pva.ScalarType.BOOLEAN], 'byteValue':
            [pva.ScalarType.BYTE], 'shortValue': [pva.ScalarType.SHORT], 'intValue':
            [pva.ScalarType.INT], 'longValue': [pva.ScalarType.LONG], 'ubyteValue':
            [pva.ScalarType.UBYTE], 'ushortValue': [pva.ScalarType.USHORT], 'uintValue':
            [pva.ScalarType.UINT], 'ulongValue': [pva.ScalarType.ULONG], 'floatValue':
            [pva.ScalarType.FLOAT], 'doubleValue': [pva.ScalarType.DOUBLE]},), 'codec':
            {'name': pva.ScalarType.STRING, 'parameters': ()}, 'compressedSize':
            pva.ScalarType.LONG, 'uncompressedSize': pva.ScalarType.LONG, 'dimension':
            [{'size': pva.ScalarType.INT, 'offset': pva.ScalarType.INT, 'fullSize':
                pva.ScalarType.INT, 'binning': pva.ScalarType.INT, 'reverse':
                pva.ScalarType.BOOLEAN}], 'uniqueId': pva.ScalarType.INT, 'dataTimeStamp':
            {'secondsPastEpoch': pva.ScalarType.LONG, 'nanoseconds': pva.ScalarType.INT,
             'userTag': pva.ScalarType.INT}, 'attribute':
            [{'name': pva.ScalarType.STRING, 'value': (), 'descriptor': pva.ScalarType.STRING,
              'sourceType': pva.ScalarType.INT, 'source': pva.ScalarType.STRING}], 'descriptor':
            pva.ScalarType.STRING, 'alarm': {'severity': pva.ScalarType.INT, 'status':
            pva.ScalarType.INT, 'message': pva.ScalarType.STRING}, 'timeStamp':
            {'secondsPastEpoch': pva.ScalarType.LONG, 'nanoseconds': pva.ScalarType.INT, 'userTag':
                pva.ScalarType.INT}, 'display': {'limitLow': pva.ScalarType.DOUBLE, 'limitHigh':
            pva.ScalarType.DOUBLE, 'description': pva.ScalarType.STRING, 'format':
            pva.ScalarType.STRING, 'units': pva.ScalarType.STRING}}

        self.pv_rec = pva.PvObject(pva_image_dict)
        self.pvaServer = pva.PvaServer(channel_name, self.pv_rec)
        self.Qchannel_name.setText(channel_name)
        self.pvaServer.start()

        self.reco_rec = pva.PvObject(pva_image_dict)
        self.pvaServer_rec = pva.PvaServer(channel_name_rec, self.reco_rec)
        self.pvaServer_rec.start()

        self.proj_rec = pva.PvObject(pva_image_dict)
        self.pvaServer_proj = pva.PvaServer(channel_name_proj, self.proj_rec)
        self.pvaServer_proj.start()

        self.omega_pv = epics.PV("acsMotion:m2.RBV")
        self.starting_angle = self.omega_pv.get()
        self.jogF_pv = epics.PV("acsMotion:m2.JOGF", auto_monitor=True)
        self.start_rotate.clicked.connect(self.rotate)
        self.add_table_button.clicked.connect(self.add_to_table)
        self.save_table_button.clicked.connect(self.save_table)

        self.piezo45_pv = epics.PV("MCS2Hex:CTm2.RBV")
        self.piezo135_pv = epics.PV("MCS2Hex:CTm1.RBV")
        self.distance_pv = epics.PV("faulhaber:m1.RBV", auto_monitor=True)
        self.lens_pv = epics.PV("OMS58:25009007_MnuAct.SVAL")
        self.exp_time_pv = epics.PV("PCOEdge:cam1:AcquirePeriod_RBV", auto_monitor=True)
        self.W_velocity_pv = epics.PV("acsMotion:m2.VELO", auto_monitor=True)

        self.sizeX_pv = epics.PV("PCOEdge:ROI1:ArraySizeX_RBV", auto_monitor=True)
        self.sizeY_pv = epics.PV("PCOEdge:ROI1:ArraySizeY_RBV", auto_monitor=True)
        self.binningx_pv = epics.PV("PCOEdge:ROI1:BinX_RBV", auto_monitor=True)
        self.binningy_pv = epics.PV("PCOEdge:ROI1:BinY_RBV", auto_monitor=True)

        self.full_sizeX_pv = epics.PV("PCOEdge:ROI2:ArraySizeX_RBV", auto_monitor=True)
        self.full_sizeY_pv = epics.PV("PCOEdge:ROI2:ArraySizeY_RBV", auto_monitor=True)
        self.proj_size_x = int(self.full_sizeX_pv.get())
        self.proj_size_y = int(self.full_sizeY_pv.get())

        self.piezo45_pv_forward = epics.PV("MCS2Hex:CTm2.TWF")
        self.piezo45_pv_backward = epics.PV("MCS2Hex:CTm2.TWR")
        self.piezo45_pv_value = epics.PV("MCS2Hex:CTm2.TWV")

        self.piezo135_pv_forward = epics.PV("MCS2Hex:CTm1.TWR")
        self.piezo135_pv_backward = epics.PV("MCS2Hex:CTm1.TWF") #these 2 are flipped on purpose, so that the +/- logic is the same on both 45/135
        self.piezo135_pv_value = epics.PV("MCS2Hex:CTm1.TWV")

        self.piezo_45_plus.clicked.connect(lambda: self.put_values(pv =self.piezo45_pv_forward, value=1))
        self.piezo_45_minus.clicked.connect(lambda: self.put_values(pv =self.piezo45_pv_backward, value=1))

        self.piezo_135_plus.clicked.connect(lambda: self.put_values(pv =self.piezo135_pv_forward, value=1))
        self.piezo_135_minus.clicked.connect(lambda: self.put_values(pv =self.piezo135_pv_backward, value=1))

        # Angular acquisition/reconstruction validity state
        self.piezo_dmov = {
            "MCS2Hex:CTm1.DMOV": None,
            "MCS2Hex:CTm2.DMOV": None,
            "OMS58:25008001.DMOV":None,
            "faulhaber:m1.DMOV":None,
            "PEGAS:miocb0102001.DMOV":None,
        }
        self.parameters_changed = True
        self.full_rotation = False
        self.stable_projection_count = 0
        self.last_lens = None

        self.binningx, self.binningy = self.binningx_pv.get(), self.binningy_pv.get()
        self.binning.setValue(self.binningx)

        self.camera_acquire_pv = epics.PV("PCOEdge:cam1:Acquire")
        self.camera_acquiretime_pv = epics.PV("PCOEdge:cam1:AcquireTime")
        self.camera_acquireperiod_pv = epics.PV("PCOEdge:cam1:AcquirePeriod")


        self.sizeX, self.sizeY = int(self.sizeX_pv.get()), int(self.sizeY_pv.get())

        self.ringbuffer_exists = 0

        initial_config = self._read_acquisition_config()
        if initial_config is None:
            raise RuntimeError('Acquisition PVs are unavailable or contain invalid values')
        self._set_acquisition_config(initial_config)
        self.acquisition_snapshot = initial_config

        self.ringbuffer_projection_size = int(math.ceil(self.ringbuffer_size[0] / 10))
        old_dial = self.dial
        self.dial = AngularStatusWidget(
            self.ringbuffer_size[0],
            self.ringbuffer_projection_size,
            old_dial.parentWidget(),
        )
        self.dial.setObjectName('dial')
        self.gridLayout_6.replaceWidget(old_dial, self.dial)
        old_dial.deleteLater()

        self.piezo_dmov_pvs = {}
        for pvname in self.piezo_dmov:
            monitored_pv = epics.PV(pvname, auto_monitor=True)
            self.piezo_dmov_pvs[pvname] = monitored_pv
            monitored_pv.add_callback(self.pv_monitor, run_now=True)

        self.prefill_CORs()

        self.COR_2x_flag = False
        self.COR_5x_flag = False
        self.COR_10x_flag = False
        self.COR_20x_flag = False

        self.lens_config = {
            '2x': (3.6, self.COR_1, self.spinBox_ruler_grid_1, 'COR_2x_flag'),
            '5x': (1.44, self.COR_2, self.spinBox_ruler_grid_2, 'COR_5x_flag'),
            '10x': (0.72, self.COR_3, self.spinBox_ruler_grid_3, 'COR_10x_flag'),
            '20x': (0.36, self.COR_4, self.spinBox_ruler_grid_4, 'COR_20x_flag'),
        }
        for lens_name, (_, _, ruler_control, _) in self.lens_config.items():
            ruler_control.valueChanged.connect(
                lambda value, lens=lens_name: self._set_piezo_step(lens, value)
            )

        self.update_pixel_size()

        self.i = 0

        self.image_pv = epics.PV("PCOEdge:image1:ArrayData", auto_monitor=True)
        self.proj_pv = epics.PV("PCOEdge:image2:ArrayData", auto_monitor=True)

        self.roi1_minY_pv = epics.PV("PCOEdge:ROI1:MinY_RBV")
        self.roi2_biny_pv = epics.PV("PCOEdge:ROI2:BinY_RBV")
        self.roi1_minY = self.roi1_minY_pv.get()
        self.roi2_biny = self.roi2_biny_pv.get()

        self.starting_omega_pv = self.omega_pv.get()
        self.projection_lock = threading.Lock()
        self.dial.selection_changed.connect(self.send_projection)
        self._allocate_live_buffers()

        self.ruler_grid_line_thickness = 1
        self.rotation_offset = 45 #still under question
        self.label_x = 'Piezo 45 [um]'
        self.label_y = 'Piezo 135 [um]'

        self.ui_size_x = self.sizeX
        self.jogF_pv.add_callback(self._rotation_pv_callback, run_now=True)
        self.distance_pv.add_callback(self._distance_pv_callback, run_now=True)
        self.image_pv.add_callback(self.update)

    def _rotation_pv_callback(self, value=None, **kwargs):
        self.rotation_state_received.emit(value)

    def _distance_pv_callback(self, value=None, **kwargs):
        self.distance_received.emit(value)

    @QtCore.pyqtSlot(object)
    def _apply_distance(self, value):
        if value is None:
            self.doubleSpinBox_distance_2.clear()
            return
        try:
            self.doubleSpinBox_distance_2.setValue(float(value))
        except (TypeError, ValueError, OverflowError):
            self.doubleSpinBox_distance_2.clear()

    @QtCore.pyqtSlot(object)
    def _apply_rotation_state(self, value):
        if value is None:
            self.rotation_running = False
            self.start_rotate.setText('Rotation unavailable')
            self.start_rotate.setEnabled(False)
            return

        self.rotation_running = bool(round(float(value)))
        self.start_rotate.setText('STOP' if self.rotation_running else 'Start endless rotation')
        self.start_rotate.setEnabled(True)

    def _put_checked(self, pv, value):
        result = pv.put(value)
        time.sleep(0.5)
        if result != 1:
            raise RuntimeError('Could not write {} to {}'.format(value, pv.pvname))

    def _refresh_rotation_state(self):
        self.rotation_state_received.emit(self.jogF_pv.get(use_monitor=True))

    def rotate(self):
        print('rotate function called')
        rotate_status = self.jogF_pv.get(use_monitor=True)
        print('rotate_status {}'.format(rotate_status))
        if rotate_status is None:
            self._apply_rotation_state(None)
            return

        self.start_rotate.setEnabled(False)
        try:
            if bool(round(float(rotate_status))):
                self.start_rotate.setText('Stopping...')
                self._put_checked(self.jogF_pv, 0)
            else:
                self.start_rotate.setText('Starting...')
                self._put_checked(self.camera_acquire_pv, 0)
                time.sleep(0.1)
                self._put_checked(self.camera_acquiretime_pv, 0.0175)
                time.sleep(0.1)
                self._put_checked(self.camera_acquireperiod_pv, 0.02)
                time.sleep(0.1)
                self._put_checked(self.camera_acquire_pv, 1)

                starting_angle = self.omega_pv.get()
                if starting_angle is not None:
                    self.starting_angle = starting_angle
                    self.starting_omega_pv = starting_angle
                self._put_checked(self.jogF_pv, 1)
        except (RuntimeError, TypeError, ValueError) as error:
            print('Could not change rotation state:', error)
        finally:
            QtCore.QTimer.singleShot(500, self._refresh_rotation_state)

    def add_to_table(self):
        row = self.positions_table.rowCount()
        self.positions_table.insertRow(row)

        pvs = [
            "OMS58:25008001.RBV",
            "MCS2Hex:CTm2.RBV",
            "MCS2Hex:CTm1.RBV",
            "faulhaber:m1.RBV",
        ]

        for column, pv_name in enumerate(pvs):
            value = epics.PV(pv_name).get()
            self.positions_table.setItem(
                row,
                column,
                QtWidgets.QTableWidgetItem(str(value))
            )

    def delete_selected_rows(self):
        rows = sorted(
            {index.row() for index in self.positions_table.selectedIndexes()},
            reverse=True
        )

        for row in rows:
            self.positions_table.removeRow(row)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            self.delete_selected_rows()
        else:
            super().keyPressEvent(event)

    def save_table(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Table",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filename:
            return

        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile,delimiter = ';', quotechar = ' ')

            # Write column headers
            headers = []
            for col in range(self.positions_table.columnCount()):
                item = self.positions_table.horizontalHeaderItem(col)
                headers.append(item.text() if item is not None else "")
            writer.writerow(headers)

            # Write table data
            for row in range(self.positions_table.rowCount()):
                data = []
                for col in range(self.positions_table.columnCount()):
                    item = self.positions_table.item(row, col)
                    data.append(item.text() if item is not None else "")
                writer.writerow(data)

    def dashed_vertical_line(self,img, y, x0, x1, color, stroke=5, gap=5):
        stride = stroke + gap
        for x in range(x0, x1):
            if (x % stride) < stroke:
                img[y, x] = color

    @QtCore.pyqtSlot(int)
    def send_projection(self, projection_index):
        projection_index = int(projection_index) % self.ringbuffer_projection_size
        with self.projection_lock:
            if not self.projection_valid[projection_index]:
                print('Projection slot', projection_index, 'has not been acquired yet')
                return
            self.projection = self.ringbuffer_projection[projection_index].copy()

        self.ruler_grid_color = math.ceil(numpy.max(self.projection))


        self.dashed_vertical_line(img=self.projection,y= int(self.roi1_minY/self.roi2_biny), x0=0,x1= self.projection.shape[1]-1, color= self.ruler_grid_color)

        self.proj_rec['value'] = (
            {'floatValue': self.projection.flatten().astype(numpy.float32)},
        )
        print(
            'Projection sent:',
            projection_index,
            'angle:',
            round(projection_index * 360.0 / self.ringbuffer_projection_size, 1),
        )


    def put_values(self, pv, value):
        pv.put(value)

    def _read_acquisition_config(self):
        values = {
            'period': self.exp_time_pv.get(use_monitor=True),
            'velocity': self.W_velocity_pv.get(use_monitor=True),
            'size_x': self.sizeX_pv.get(use_monitor=True),
            'size_y': self.sizeY_pv.get(use_monitor=True),
            'bin_x': self.binningx_pv.get(use_monitor=True),
            'bin_y': self.binningy_pv.get(use_monitor=True),
            'proj_x': self.full_sizeX_pv.get(use_monitor=True),
            'proj_y': self.full_sizeY_pv.get(use_monitor=True),
        }
        if any(value is None for value in values.values()):
            return None

        try:
            config = {
                'period': round(float(values['period']), 9),
                'velocity': round(abs(float(values['velocity'])), 9),
                'size_x': int(values['size_x']),
                'size_y': int(values['size_y']),
                'bin_x': int(values['bin_x']),
                'bin_y': int(values['bin_y']),
                'proj_x': int(values['proj_x']),
                'proj_y': int(values['proj_y']),
            }
        except (TypeError, ValueError, OverflowError):
            return None

        if (
            config['period'] <= 0
            or config['velocity'] <= 0
            or min(config['size_x'], config['size_y'], config['proj_x'], config['proj_y']) <= 0
            or min(config['bin_x'], config['bin_y']) <= 0
        ):
            return None
        return config

    def _set_acquisition_config(self, config):
        self.sizeX = config['size_x']
        self.sizeY = config['size_y']
        self.binningx = config['bin_x']
        self.binningy = config['bin_y']
        self.proj_size_x = config['proj_x']
        self.proj_size_y = config['proj_y']
        projection_count = max(2, round(360.0 / (config['velocity'] * config['period'])))
        self.ringbuffer_size = (projection_count, 1, self.sizeX)
        self.ringbuffer_projection_size = int(math.ceil(projection_count / 10))

    def _allocate_live_buffers(self):
        self.ringbuffer = numpy.ones(self.ringbuffer_size, dtype='H')
        self.sino_chopped = numpy.zeros(
            (int(self.ringbuffer_size[0] / 2), 1, self.ringbuffer_size[2]),
            dtype='H',
        )
        with self.projection_lock:
            self.ringbuffer_projection = numpy.zeros(
                (self.ringbuffer_projection_size, self.proj_size_y, self.proj_size_x),
                dtype=numpy.float32,
            )
            self.projection_valid = numpy.zeros(self.ringbuffer_projection_size, dtype=bool)

        self.pv_rec['dimension'] = [
            {'size': self.sizeX, 'fullSize': self.sizeX, 'binning': 1},
            {'size': int(self.ringbuffer_size[0]), 'fullSize': int(self.ringbuffer_size[0]), 'binning': 1},
        ]
        self.proj_rec['dimension'] = [
            {'size': self.proj_size_x, 'fullSize': self.proj_size_x, 'binning': 1},
            {'size': self.proj_size_y, 'fullSize': self.proj_size_y, 'binning': 1},
        ]
        self.ringbuffer_exists = 1
        print('ringbuffers created:', self.ringbuffer.shape, self.ringbuffer_projection.shape)

    def check_parameters(self):
        config = self._read_acquisition_config()
        if config is None or config == self.acquisition_snapshot:
            return False

        previous_size_x = self.sizeX
        print('Acquisition parameters changed:', self.acquisition_snapshot, '->', config)
        self._set_acquisition_config(config)
        self._allocate_live_buffers()
        self.acquisition_snapshot = config
        self.i = 0

        current_angle = self.omega_pv.get(use_monitor=True)
        if current_angle is not None:
            self.starting_angle = current_angle
            self.starting_omega_pv = current_angle

        roi1_min_y = self.roi1_minY_pv.get(use_monitor=True)
        roi2_bin_y = self.roi2_biny_pv.get(use_monitor=True)
        if roi1_min_y is not None:
            self.roi1_minY = roi1_min_y
        if roi2_bin_y is not None:
            self.roi2_biny = roi2_bin_y

        self.parameters_have_changed()
        self.acquisition_config_changed.emit(
            int(self.ringbuffer_size[0]),
            self.ringbuffer_projection_size,
            self.sizeX,
            self.binningx,
        )
        return previous_size_x != self.sizeX

    @QtCore.pyqtSlot(int, int, int, int)
    def _apply_acquisition_config_ui(self, projection_count, selector_count, size_x, binning_x):
        self.dial.set_counts(projection_count, selector_count)
        self.binning.setValue(binning_x)

        if size_x != self.ui_size_x:
            center_shift = (size_x - self.ui_size_x) / 2.0
            for cor_control in (self.COR_1, self.COR_2, self.COR_3, self.COR_4):
                cor_control.setValue(cor_control.value() + center_shift)
            self.ui_size_x = size_x

        self.update_pixel_size()


    def parameters_have_changed(self, moving=False):
        """Invalidate the current ringbuffer and reconstruction."""
        self.parameters_changed = True
        self.full_rotation = False
        self.stable_projection_count = 0
        self.dial.reset_history()

    def pv_monitor(self, pvname=None, value=None, **kwargs):
        try:
            if pvname not in self.piezo_dmov or value is None:
                return

            value = int(float(value))
            self.piezo_dmov[pvname] = value

            # Either piezo is moving.
            if any(dmov == 0 for dmov in self.piezo_dmov.values()):
                self.parameters_have_changed(moving=True)
                return

            # Wait until both initial monitor values are known.
            if None in self.piezo_dmov.values():
                return

            # Both piezos are stopped. Keep the reconstruction invalid
            # until a complete fresh rotation has been collected.
            if all(dmov == 1 for dmov in self.piezo_dmov.values()):
                return

        except (ValueError, TypeError, AttributeError) as error:
            print("Could not process DMOV monitor:", pvname, value, error)


    def update(self, **kwargs):
        # Camera acquisition may remain active while the rotation axis is stopped.
        # Those frames have no new angle and must not enter the tomography buffers.
        if not self.rotation_running:
            return

        rawimgflat = self.image_pv.get()
        if rawimgflat is None:
            return

        if (self.i % 25) == 0:
            self.check_parameters()

        rawimgflat = numpy.asarray(rawimgflat)
        expected_image_size = self.sizeX * self.sizeY
        if rawimgflat.size != expected_image_size:
            self.check_parameters()
            expected_image_size = self.sizeX * self.sizeY
            if rawimgflat.size != expected_image_size:
                print('Image size mismatch:', rawimgflat.size, 'expected:', expected_image_size)
                return

        self.ringbuffer[self.i % self.ringbuffer_size[0],:,:] = rawimgflat[-(round(self.sizeY / 2)) * self.sizeX: -(round(self.sizeY / 2) -1) * self.sizeX]

        # While the current parameters are invalid, count new projections.
        # A full ringbuffer refill represents one complete fresh rotation.
        piezo_state_known = None not in self.piezo_dmov.values()
        both_piezos_stopped = (
            piezo_state_known
            and all(dmov == 1 for dmov in self.piezo_dmov.values())
        )
        piezo_is_moving = (
            piezo_state_known
            and any(dmov == 0 for dmov in self.piezo_dmov.values())
        )

        acquisition_angle = (
            float(self.starting_angle or 0.0)
            + (self.i % self.ringbuffer_size[0]) * 360.0 / self.ringbuffer_size[0]
        ) % 360.0
        self.dial.record_projection(acquisition_angle, piezo_is_moving)

        if self.parameters_changed and not self.full_rotation and both_piezos_stopped:
            self.stable_projection_count += 1
            required_projections = int(math.ceil(self.ringbuffer_size[0] / 2))

            if self.stable_projection_count >= required_projections:
                self.full_rotation = True
                self.dial.mark_stable_complete()
                print('Stable 180 degree rotation collected')

        self.current_omega_pv = self.omega_pv.get()

        if (self.i % 10) == 0:
            projection = self.proj_pv.get()
            if projection is not None:
                projection = numpy.asarray(projection)
                expected_size = self.proj_size_x * self.proj_size_y
                if projection.size == expected_size:
                    projection_slot = int((self.i % self.ringbuffer_size[0]) // 10)
                    projection_frame = projection.reshape((self.proj_size_y, self.proj_size_x))
                    with self.projection_lock:
                        self.ringbuffer_projection[projection_slot, :, :] = projection_frame
                        self.projection_valid[projection_slot] = True

                    # Keep the selected view current when its slot is refreshed.
                    if projection_slot == self.dial.selected_projection:
                        self.send_projection(projection_slot)
                else:
                    print(
                        'Projection size mismatch:',
                        projection.size,
                        'expected:',
                        expected_size,
                    )

        sinogram = self.ringbuffer[:, 0, :]
        self.pv_rec['dimension'] = [
            {'size': int(sinogram.shape[1]), 'fullSize': int(sinogram.shape[1]), 'binning': 1},
            {'size': int(sinogram.shape[0]), 'fullSize': int(sinogram.shape[0]), 'binning': 1}
        ]
        self.pv_rec['value'] = (
            {'floatValue': sinogram.flatten().astype(numpy.float32)},
        )
        #print(self.current_omega_pv, 'current omega')
        if (self.i % 5) == 0:
            print('FEEDING IMAGE')
            sinogram = self.ringbuffer[:,0,:]
            #imgplotted.set_data(sinogram)
            #angles.set_data(float(self.omega_pv.get()) % 360, self.i)
            #fig.canvas.flush_events()
            #self.slice_show = sinogram.astype(numpy.float32)
            #print('before read param')

            self.refresh_ui_requested.emit()
            #print('after read param')

            position = self.i % self.ringbuffer_size[0]
            print('position', position)
            # if position >= int((self.ringbuffer_size[0]/2)-1):
            #     self.sino_chopped[:,0,:] = sinogram[position-int(self.ringbuffer_size[0]/2):position,:]
            # else:
            #     self.sino_chopped[-position-1:,0,:] = sinogram[:position+1,:]
            #     self.sino_chopped[:int(self.ringbuffer_size[0]/2)-position,0,:] = sinogram[-int(self.ringbuffer_size[0]/2) + position:,:]

            half = int(self.ringbuffer_size[0] / 2)
            N = int(self.ringbuffer_size[0])

            idx = (numpy.arange(position - half + 1, position + 1) % N).astype(int)

            self.sino_chopped[:, 0, :] = sinogram[idx, :]

            self.sino_chopped = self.sino_chopped.astype(numpy.float32)
            # write result to pv
            #self.pv_rec['value'] = ({'floatValue': self.sino_chopped.flatten()},)
            self.pv_rec['value'] = ({'floatValue': sinogram.flatten().astype(numpy.float32)},)

            #plt.imshow(sinogram, cmap='gray')
            #plt.show()
            print(int(self.ringbuffer_size[0]/2))
            self.extended_sinos = tomopy.minus_log(self.sino_chopped)
            self.extend_FOV_fixed_ImageJ_Stream = 0.5
            self.full_size = self.extended_sinos.shape[2]
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            #+self.current_omega_pv/180*math.pi
            self.extended_sinos = tomopy.misc.morph.pad(self.extended_sinos, axis=2,
                                                   npad=round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size),
                                                   mode='edge')
            print('extended sinos size',self.extended_sinos.shape)



            self.slice = tomopy.recon(self.extended_sinos[:,:,:], numpy.linspace(0,math.pi,int(self.ringbuffer_size[0]/2),endpoint=False)+(((self.i % self.ringbuffer_size[0])/self.ringbuffer_size[0]))*2*math.pi + self.starting_omega_pv/180*math.pi - self.rotation_offset/180*math.pi,
                                      center=float(self.COR)+round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size)
                                      ,options=options,algorithm=tomopy.astra)

            self.slice = tomopy.circ_mask(self.slice, axis=0, ratio=1.0, val=-1)
            print(self.slice.shape)
            self.slice = self.slice[0,:,:]
            print(self.slice.shape)


            if self.enable_grid.isChecked() == True:
                self.slice = self.add_ruler()

            self.reco_rec['dimension'] = [
                {'size': int(self.slice.shape[1]), 'fullSize': int(self.slice.shape[1]), 'binning': 1},
                {'size': int(self.slice.shape[0]), 'fullSize': int(self.slice.shape[0]), 'binning': 1}
            ]
            self.reco_rec['value'] = (
                {'floatValue': self.slice.flatten().astype(numpy.float32)},
            )

            # The published reconstruction is valid only after one complete
            # rotation has been acquired with unchanged parameters.
            if self.parameters_changed and self.full_rotation:
                self.parameters_changed = False
                print('Stable reconstruction published')

        #print('i', self.i, 'Modulus:', self.i % self.ringbuffer_size[0])
        self.i = self.i +1

    def read_parameter(self):
        self.update_pixel_size()


    def buttons_deactivate_all(self):
        self.COR_1.setEnabled(False)
        self.COR_2.setEnabled(False)
        self.COR_3.setEnabled(False)
        self.COR_4.setEnabled(False)

        self.COR_2x_flag = False
        self.COR_5x_flag = False
        self.COR_10x_flag = False
        self.COR_20x_flag = False

    def prefill_CORs(self):

        print('preset CORs to half image size')
        self.COR_1.setValue(round(self.sizeX_pv.get() / 2))
        self.COR_2.setValue(round(self.sizeX_pv.get() / 2))
        self.COR_3.setValue(round(self.sizeX_pv.get() / 2))
        self.COR_4.setValue(round(self.sizeX_pv.get() / 2))

    def update_pixel_size(self):
        current_lens = self.lens_pv.get()
        if current_lens not in self.lens_config:
            self.buttons_deactivate_all()
            print('Unknown or unavailable lens:', current_lens)
            return

        lens_changed = current_lens != self.last_lens
        if self.last_lens is not None and lens_changed:
            print('Optics changed:', self.last_lens, '->', current_lens)
            piezo_is_moving = any(
                dmov == 0 for dmov in self.piezo_dmov.values()
            )
            self.parameters_have_changed(moving=piezo_is_moving)

        self.last_lens = current_lens
        pixel_size, cor_control, ruler_control, active_flag = self.lens_config[current_lens]
        if not getattr(self, active_flag):
            self.buttons_deactivate_all()
            cor_control.setEnabled(True)
            setattr(self, active_flag, True)
            self._set_piezo_step(current_lens, ruler_control.value(), force=True)

        self.pixel_size_set = pixel_size
        self.COR = cor_control.value()
        self.spinBox_ruler_grid = ruler_control.value()
        self.pixel_size.setValue(pixel_size)

    def _set_piezo_step(self, lens_name, value, force=False):
        if not force and lens_name != self.last_lens:
            return
        step_size = value / 1000.0
        self.piezo45_pv_value.put(step_size)
        self.piezo135_pv_value.put(step_size)

    def add_ruler(self):


        self.piezo45_val = -self.piezo45_pv.get()

        self.piezo135_val = self.piezo135_pv.get()

        if str(self.piezo45_val) == 'None':
            self.piezo_45_proxy = 0
        else:
            self.piezo_45_proxy = self.piezo45_val
        print('piezo 45', self.piezo_45_proxy)

        if str(self.piezo135_val) == 'None':
            self.piezo_135_proxy = 0
        else:
            self.piezo_135_proxy = self.piezo135_val
        print('piezo 135', self.piezo_135_proxy)

        self.ruler_grid_color = math.ceil(numpy.max(self.slice))

        # draws a circle with the detector size as diameter
        cv2.circle(self.slice, (round(self.slice.shape[1] / 2), round(self.slice.shape[1] / 2)), round(self.sizeX/2), self.ruler_grid_color, self.ruler_grid_line_thickness)

        # add label x
        cv2.putText(self.slice, self.label_x, (
        round(self.slice.shape[1] / 2) + 20, round(self.ruler_grid_line_thickness) * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                    thickness=self.ruler_grid_line_thickness)

        # add ruler +X
        for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value() / 2),
                       round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value()),
                       round(self.spinBox_ruler_grid)):

            cv2.line(self.slice, (round(r / (self.binning.value() * self.pixel_size.value())), 0),
                     (round(r / (self.pixel_size.value() * self.binning.value())), self.slice.shape[0]), self.ruler_grid_color, self.ruler_grid_line_thickness)
            cv2.putText(self.slice, str(-(round(1000 * self.piezo_45_proxy/  5) * 5   +   round( r / 5) * 5   -   round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5)),
                        (round(r / (self.pixel_size.value() * self.binning.value())) + 20, round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                        self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

        # add ruler -X
        for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value() / 2), 0,
                       -round(self.spinBox_ruler_grid)):
            cv2.line(self.slice, (round(r / (self.pixel_size.value() * self.binning.value())), 0),
                     (round(r / (self.pixel_size.value() * self.binning.value())), self.slice.shape[0]), self.ruler_grid_color, self.ruler_grid_line_thickness)
            cv2.putText(self.slice, str(-(round(1000 * self.piezo_45_proxy / 5) * 5  +  round(
                r  / 5) * 5  -  round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5)),
                        (round(r / (self.pixel_size.value() * self.binning.value() )) + 20, round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                        self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)


        # add label Y
        cv2.putText(self.slice, self.label_y, (20, round(self.slice.shape[1] / 2) -40),
                    cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                    thickness=self.ruler_grid_line_thickness)

        # add ruler +Y
        for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value() / 2),
                       round(self.slice.shape[1] * self.pixel_size.value()* self.binning.value()),
                       round(self.spinBox_ruler_grid)):
            cv2.line(self.slice, (0, round(r / (self.pixel_size.value()* self.binning.value()))),
                     (self.slice.shape[1], round(r / (self.pixel_size.value()* self.binning.value()))), self.ruler_grid_color, self.ruler_grid_line_thickness)
            cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy / 5) * 5  +  round(
                r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5),
                        (20, round(r / (self.pixel_size.value() * self.binning.value())) + round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                        self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

        # add ruler -Y
        for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value()/ 2), 0,
                       -round(self.spinBox_ruler_grid)):
            cv2.line(self.slice, (0, round(r / (self.pixel_size.value()* self.binning.value()))),
                     (self.slice.shape[1], round(r / (self.pixel_size.value()* self.binning.value()))), self.ruler_grid_color, self.ruler_grid_line_thickness)
            cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy / 5) * 5 + round(
                r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5),
                        (20, round(r / (self.pixel_size.value()* self.binning.value())) + round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                        self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

        return self.slice




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = OnTheFlyNavigator()
    main.show()
    sys.exit(app.exec_())

#end of code
