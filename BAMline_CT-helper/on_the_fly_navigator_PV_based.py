from os import wait

import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
from PIL import Image
import h5py
import tomopy
import math
import time
import os
import csv
import threading
import cv2                                      #to install package with pycharm search for "opencv-python"
from scipy.ndimage.filters import gaussian_filter, median_filter
import pvaccess as pva                          #to install package with pycharm search for "pvapy"
import epics
import matplotlib.pyplot as plt
#import Image



# On-the-fly Navigator
version =  "Version 2026.07.12"

os.environ["EPICS_CA_ADDR_LIST"] = "172.31.20.131 172.31.20.231 172.31.20.145"

#Install ImageJ-PlugIn: EPICS AreaDetector NTNDA-Viewer, look for the channel specified here under channel_name, consider multiple users on servers!!!
channel_name = 'BAMline:Navigator'
channel_name_rec = 'BAMline:NavigatorReco'
channel_name_proj = 'BAMline:NavigatorProj'

#standard_path = "C:/temp/HDF5-Reading/220130_1734_604_J1_anode_half_cell_in-situ_Z30_Y5430_15000eV_1p44um_500ms/" # '/mnt/raid/CT/2022/'
standard_path = r'C:/delete/reg_data/18_230606_2044_AlTi_F_Ref_tomo___Z25_Y6500_25000eV_10x_400ms'

Ui_on_the_fly_Navigator_Window, Q_on_the_fly_Navigator_Window = loadUiType('on_the_fly_navigator.ui')  # connect to the GUI for the program

#plt.ion()
#fig,(ax,ax2) = plt.subplots(2,1)
#imgplotted = ax.imshow(numpy.zeros((6000,2560)),cmap='gray',vmin=0,vmax=16384)
#imgplotted = ax.imshow(numpy.zeros((720,2560)),cmap='gray',vmin=0,vmax=65535)
#angles, = ax2.plot(0,0,'o',color='red', )

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


    def __init__(self):
        super(OnTheFlyNavigator, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('On-the-fly Navigator')


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
        self.jogF_pv = epics.PV("acsMotion:m2.JOGF")
        self.jofvelo_pv = epics.PV("acsMotion:m2.JVEL")
        self.omega_stop_pv = epics.PV("acsMotion:m2.STOP")
        self.start_rotate.clicked.connect(self.rotate)

        self.piezo45_pv = epics.PV("MCS2Hex:CTm2.RBV")
        self.piezo135_pv = epics.PV("MCS2Hex:CTm1.RBV")
        self.energy_pv = epics.PV("Energ:25000007rbv")
        self.distance_pv = epics.PV("faulhaber:m1.RBV")
        self.lens_pv = epics.PV("OMS58:25009007_MnuAct.SVAL")
        self.exp_time_pv =epics.PV("PCOEdge:cam1:AcquirePeriod_RBV")
        self.W_velocity_pv = epics.PV("acsMotion:m2.VELO")

        pv_cam_exp = epics.PV("PCOEdge:cam1:AcquireTime")
        pv_cam_period = epics.PV("PCOEdge:cam1:AcquirePeriod")

        self.sizeX_pv = epics.PV("PCOEdge:ROI1:ArraySizeX_RBV")
        self.sizeY_pv = epics.PV("PCOEdge:ROI1:ArraySizeY_RBV")
        self.binningx_pv = epics.PV("PCOEdge:ROI1:BinX_RBV")
        self.binningy_pv = epics.PV("PCOEdge:ROI1:BinY_RBV")

        self.full_sizeX_pv = epics.PV("PCOEdge:ROI2:ArraySizeX_RBV")
        self.full_sizeY_pv = epics.PV("PCOEdge:ROI2:ArraySizeY_RBV")
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

        self.binningx,self.binningy = self.binningx_pv.get(),self.binningy_pv.get()
        self.binning.setValue(self.binningx)

        self.camera_acquire_pv = epics.PV("PCOEdge:cam1:Acquire")
        self.camera_acquiretime_pv = epics.PV("PCOEdge:cam1:AcquireTime")
        self.camera_acquireperiod_pv = epics.PV("PCOEdge:cam1:AcquirePeriod")


        #self.sizeX, self.sizeY = round(self.sizeX_pv.get()/self.binningx), round(self.sizeY_pv.get()/self.binningy)
        #self.sizeX, self.sizeY = int(self.sizeX_pv.get()/self.binningx_pv.get()), int(self.sizeY_pv.get()/self.binningy_pv.get())
        self.sizeX, self.sizeY = int(self.sizeX_pv.get()), int(self.sizeY_pv.get())

        self.ringbuffer_exists = 0

        if self.W_velocity_pv.get() != 0:
            print('velocity: ', self.W_velocity_pv.get(), '     exp time: ', self.exp_time_pv.get(), '      size X: ', self.sizeX_pv.get())
            self.ringbuffer_size = (
            round(360 / (self.W_velocity_pv.get() * self.exp_time_pv.get())), 1, self.sizeX_pv.get())
            print('ringbuffer_size', self.ringbuffer_size)
        else:
            print('No rotation detected!')

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

        self.update_pixel_size()

        self.i = 0

        self.image_pv = epics.PV("PCOEdge:image1:ArrayData", auto_monitor=True)
        self.proj_pv = epics.PV("PCOEdge:image2:ArrayData", auto_monitor=True)

        self.pv_rec['dimension'] = [
             {'size': self.ringbuffer_size[2], 'fullSize': self.ringbuffer_size[2], 'binning': 1},
             {'size': int(self.ringbuffer_size[0]), 'fullSize': int(self.ringbuffer_size[0]), 'binning': 1}]

        self.proj_rec['dimension'] = [
             {'size': self.proj_size_x, 'fullSize': self.proj_size_x, 'binning': 1},
             {'size': self.proj_size_y, 'fullSize': self.proj_size_y, 'binning': 1}]
        self.projection = self.proj_pv.get()

        self.roi1_minY = epics.PV("PCOEdge:ROI1:MinY_RBV").get()
        self.roi2_biny = epics.PV("PCOEdge:ROI2:BinY_RBV").get()

        self.ringbuffer = numpy.ones(self.ringbuffer_size, dtype='H')
        self.starting_omega_pv = self.omega_pv.get()
        self.ringbuffer_Micos_W = numpy.zeros(self.ringbuffer_size[0], dtype=numpy.float32)
        self.ringbuffer_projection = numpy.zeros(
            (self.ringbuffer_projection_size, self.proj_size_y, self.proj_size_x),
            dtype=numpy.float32,
        )
        self.projection_valid = numpy.zeros(self.ringbuffer_projection_size, dtype=bool)
        self.projection_lock = threading.Lock()
        self.dial.selection_changed.connect(self.send_projection)

        self.ringbuffer_exists = 1
        print('ringbuffer created with size: ', self.ringbuffer.shape)

        self.sino_chopped = numpy.zeros((int(self.ringbuffer_size[0]/2),1, self.ringbuffer_size[2]), dtype='H')

        self.ruler_grid_line_thickness = 1
        self.rotation_offset = 45 #still under question
        self.label_x = 'Piezo 45 [um]'
        self.label_y = 'Piezo 135 [um]'

        self.N = int(self.ringbuffer_size[0])
        self.start_rotate.setEnabled(True)
        self.image_pv.add_callback(self.update)

    def rotate(self):
        self.rotate_status = self.jogF_pv.get()
        if self.rotate_status == 0:
            self.camera_acquire_pv.put(0)
            time.sleep(2)
            self.camera_acquiretime_pv.put(0.0175)
            time.sleep(0.5)
            self.camera_acquireperiod_pv.put(0.02)
            time.sleep(0.1)
            self.camera_acquire_pv.put(1)
            self.jogF_pv.put(1)
            self.start_rotate.setText('STOP')
        else:
            self.jogF_pv.put(0)
            self.start_rotate.setText('Start endless rotation')
            self.starting_omega_pv = self.omega_pv.get()

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

        print('olalal')
        print(int(self.roi1_minY/self.roi2_biny))

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

    def check_changes(self):
        if self.i==0:
            return


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


    def create_ringbuffer(self):
        self.ringbuffer = numpy.ones(self.ringbuffer_size, dtype='H')
        self.ringbuffer_Micos_W = numpy.zeros(self.ringbuffer_size[0], dtype=numpy.float32)
        self.ringbuffer_projection_size = int(math.ceil(self.ringbuffer_size[0] / 10))
        self.ringbuffer_projection = numpy.zeros(
            (self.ringbuffer_projection_size, self.proj_size_y, self.proj_size_x),
            dtype=numpy.float32,
        )
        self.projection_valid = numpy.zeros(self.ringbuffer_projection_size, dtype=bool)

        self.proj_rec['dimension'] = [
            {'size': self.proj_size_x, 'fullSize': self.proj_size_x, 'binning': 1},
            {'size': self.proj_size_y, 'fullSize': self.proj_size_y, 'binning': 1}
        ]

        self.ringbuffer_exists = 1
        print('ringbuffer created with size: ', self.ringbuffer.shape)

        self.sino_chopped = numpy.zeros(((self.ringbuffer_size[0]/2),1, self.ringbuffer_size[2]), dtype='H')

    def update(self, **kwargs):
        rawimgflat = self.image_pv.get()

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

        #print('omega to initial omega difference', self.current_omega_pv-self.starting_omega_pv)
        #self.ringbuffer_Micos_W[self.i % self.ringbuffer_size[0]] = self.current_omega_pv

        #print('Micos_W Ringbuffer', self.current_omega_pv)
        #, self.ringbuffer_Micos_W)
        #self.progressBar.setValue(int(self.omega_pv.get() % 360))
        sinogram = self.ringbuffer[:, 0, :]
        self.pv_rec['dimension'] = [
            {'size': int(sinogram.shape[1]), 'fullSize': int(sinogram.shape[1]), 'binning': 1},
            {'size': int(sinogram.shape[0]), 'fullSize': int(sinogram.shape[0]), 'binning': 1}
        ]
        self.pv_rec['value'] = (
            {'floatValue': sinogram.flatten().astype(numpy.float32)},
        )
        print(self.current_omega_pv, 'current omega')
        if (self.i % 5) == 0:
            print('FEEDING IMAGE')
            sinogram = self.ringbuffer[:,0,:]
            #imgplotted.set_data(sinogram)
            #angles.set_data(float(self.omega_pv.get()) % 360, self.i)
            #fig.canvas.flush_events()
            #self.slice_show = sinogram.astype(numpy.float32)
            print('before read param')

            self.read_parameter()
            print('after read param')

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

        print('i', self.i, 'Modulus:', self.i % self.ringbuffer_size[0])
        self.i = self.i +1

    def cut_path_name(self):
        print('function cut_path_name')
        #analyse and cut the path in pieces and get relevant information from raw-file
        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        ni_htap = self.path_in[::-1]
        self.last_folder = self.path_in[len(ni_htap) - ni_htap.find('/') - 1 :  :1]
        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 1: 1]
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]
        print('chopped path: ',self.path_in, '  ', self.last_folder,'  ', self.namepart,'  ', self.filetype)
        self.Sample.setText(self.path_klick)

        #link a volume to the hdf-file
        self.f = h5py.File(self.path_klick, 'r', libver='latest', swmr=True)
        self.vol_proxy = self.f['/entry/data/data']
        #self.line_proxy = self.f['/entry/instrument/NDAttributes/CT_MICOS_W']
        self.line_proxy = self.f['/entry/instrument/NDAttributes/SAMPLE_MICOS_W1']

        print('raw data volume size: ', self.vol_proxy.shape)

        #self.prefill_parameter()
        #print('try to go to function check_180 ')
        self.check_180()    #this will end in check_auto_update



#===========================================================
    def buttons_activate_reco(self):
        print('function buttons_activate_reco')


    def prefill_parameter(self):
        print('function prefill_parameter')
        #self.prefill_slice_number()
        #self.get_rotation_angles()
        #self.find_rotation_start()
        #self.prefill_CORs()
        #self.prefill_pixel_size()
        #self.prefill_binning()
        #self.prefill_energy()
        #self.prefill_distance()


    def check_180(self):
        print('function check_180')
        while self.graph[-1] < 180:
            self.get_rotation_angles()
            print('waiting for sufficient data...', self.graph[-1])
            time.sleep(1)
        print("Enough data to proceed. angle:", self.graph[-1])
        self.check_auto_update()

    def check_auto_update(self):    #AUTO UPDATE ON/OFF?
        print('function check_auto_update')
        i=0
        while i < 1:
            j=0
            while self.auto_update.isChecked() == False:
                print('waiting for auto update...')
                QtWidgets.QApplication.processEvents()
                time.sleep(5)
            print('auto update requested')
            while self.auto_update.isChecked() == True:
                j = j + 1
                time_begin = time.time()
                self.read_parameter()
                time_read_parameter = time.time()
                #print('time_read_parameter',time_read_parameter - time_begin)
                QtWidgets.QApplication.processEvents()
                self.load_data()
                time_load_data = time.time()
                #print('time_load_data', time_load_data - time_begin)
                QtWidgets.QApplication.processEvents()
                self.reconstruct()
                time_reconstruct = time.time()
                print('iteration:', j, '  time:', round((time_reconstruct - time_begin)*1000)/1000, 'fps:', round(1/(time_reconstruct - time_begin)))
                QtWidgets.QApplication.processEvents()

    def read_parameter(self):
        #self.get_rotation_angles()
        self.update_pixel_size()
        #self.prefill_binning()
        #self.prefill_energy()
        #self.prefill_distance()


    def buttons_deactivate_all(self):
        #self.slice_number.setEnabled(False)
        self.COR_1.setEnabled(False)
        self.COR_2.setEnabled(False)
        self.COR_3.setEnabled(False)
        self.COR_4.setEnabled(False)

        self.COR_2x_flag = False
        self.COR_5x_flag = False
        self.COR_10x_flag = False
        self.COR_20x_flag = False

    def buttons_activate_load(self):
        print('function buttons_activate_load')


    def prefill_slice_number(self):  #get the image dimensions and prefill slice number
        self.slice_number.setMaximum(self.vol_proxy.shape[1]-1)
        self.slice_number.setMinimum(0)
        time.sleep(1)
        self.slice_number.setValue(round(self.vol_proxy.shape[1]/2))    # be careful with an infinite loop when setValue actually triggers valueChanged. Therefore, auto_update starts off
        print('Function prefill_slice_number: Set middle height as slice number:  ', self.slice_number.value())
        self.slice_number.setEnabled(True)

    def get_rotation_angles(self):
        #self.f = h5py.File(self.path_klick, 'r',libver='latest', swmr=True)
        #self.line_proxy = self.f['/entry/instrument/NDAttributes/CT_MICOS_W']
        #self.line_proxy = self.f['/entry/instrument/NDAttributes/SAMPLE_MICOS_W2']
        self.line_proxy.id.refresh()
        self.graph = numpy.array(self.line_proxy)
        print('Function get_rotation_angles: Found number of angles:  ', self.graph.shape[0], '      current angle: ', self.graph[-1])


    def prefill_CORs(self):

        print('preset CORs to half image size')
        self.COR_1.setValue(round(self.sizeX_pv.get() / 2))
        self.COR_2.setValue(round(self.sizeX_pv.get() / 2))
        self.COR_3.setValue(round(self.sizeX_pv.get() / 2))
        self.COR_4.setValue(round(self.sizeX_pv.get() / 2))

    def prefill_pixel_size(self):

        if '/entry/instrument/NDAttributes/CT_Pixelsize' in self.f:
            self.pixel_proxy = self.f['/entry/instrument/NDAttributes/CT_Pixelsize']
            self.pixel_size.setValue(self.pixel_proxy[-1])
            #print('Function prefill_pixel_size: ', self.pixel_proxy[-1])
        else:
            self.pixel_size.setValue(1)

    def update_pixel_size(self):
        current_lens = self.lens_pv.get()

        if self.last_lens is None:
            self.last_lens = current_lens
        elif current_lens != self.last_lens:
            print('Optics changed:', self.last_lens, '->', current_lens)
            self.last_lens = current_lens

            piezo_is_moving = any(
                dmov == 0 for dmov in self.piezo_dmov.values()
            )
            self.parameters_have_changed(moving=piezo_is_moving)

        if current_lens == '2x':
            self.pixel_size_set = 3.6
            if self.COR_2x_flag == False:
                self.buttons_deactivate_all()
                self.COR_1.setEnabled(True)
                self.COR_2x_flag = True
                self.COR_5x_flag = False
                self.COR_10x_flag = False
                self.COR_20x_flag = False

            self.piezo45_pv_value.put(self.spinBox_ruler_grid_1.value() / 1000)
            self.piezo135_pv_value.put(self.spinBox_ruler_grid_1.value() / 1000)

            self.COR = self.COR_1.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_1.value()
            self.pixel_size.setValue(self.pixel_size_set)
        elif current_lens == '5x':
            self.pixel_size_set = 1.44
            if self.COR_5x_flag == False:
                self.buttons_deactivate_all()
                self.COR_2.setEnabled(True)
                self.COR_5x_flag = True
                self.COR_10x_flag = False
                self.COR_20x_flag = False
                self.COR_2x_flag = False

            self.piezo45_pv_value.put(self.spinBox_ruler_grid_2.value() / 1000)
            self.piezo135_pv_value.put(self.spinBox_ruler_grid_2.value() / 1000)

            self.COR = self.COR_2.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_2.value()
            self.pixel_size.setValue(self.pixel_size_set)
        elif current_lens == '10x':
            self.pixel_size_set = 0.72
            if self.COR_10x_flag == False:
                self.buttons_deactivate_all()
                self.COR_3.setEnabled(True)
                self.COR_10x_flag = True
                self.COR_20x_flag = False
                self.COR_5x_flag = False
                self.COR_2x_flag = False

            self.piezo45_pv_value.put(self.spinBox_ruler_grid_3.value() / 1000)
            self.piezo135_pv_value.put(self.spinBox_ruler_grid_3.value() / 1000)

            self.COR = self.COR_3.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_3.value()
            self.pixel_size.setValue(self.pixel_size_set)
        elif current_lens == '20x':
            self.pixel_size_set = 0.36
            if self.COR_20x_flag == False:
                self.buttons_deactivate_all()
                self.COR_4.setEnabled(True)
                self.COR_20x_flag = True
                self.COR_5x_flag = False
                self.COR_2x_flag = False
                self.COR_10x_flag = False

            self.piezo45_pv_value.put(self.spinBox_ruler_grid_4.value() / 1000)
            self.piezo135_pv_value.put(self.spinBox_ruler_grid_4.value() / 1000)

            self.COR = self.COR_4.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_4.value()
            self.pixel_size.setValue(self.pixel_size_set)
        else:
            self.buttons_deactivate_all()

        print('pixel size: ', self.pixel_size_set)

    def prefill_binning(self):
        if '/entry/instrument/NDAttributes/Binning_X' in self.f:
            self.binning_proxy = self.f['/entry/instrument/NDAttributes/Binning_X']
            self.binning.setValue(self.binning_proxy[-1])
            print('Function prefill_binning: ', self.binning_proxy[-1])
        else:
            self.binning.setValue(1)
            print('Function prefill_binning: Not found. Set to 1')

    def prefill_energy(self):
        #self.energy_pv.get()
        self.doubleSpinBox_Energy_2.setValue(round(float(self.energy_pv.get()) * 100) / 100)
        """
        if '/entry/instrument/NDAttributes/DMM_Energy' in self.f:
            self.energy_proxy = self.f['/entry/instrument/NDAttributes/DMM_Energy']
            self.doubleSpinBox_Energy_2.setValue(round(self.energy_proxy[-1]*100)/100)
            #print('Function prefill_energy:', round(self.energy_proxy[-1]*100)/100)
        else:
            self.doubleSpinBox_Energy_2.setValue(1)
            #print('Function prefill_energy: Energy not found')
        """

    def prefill_distance(self):

        self.doubleSpinBox_distance_2.setValue(round(float(self.distance_pv.get()) + 25))
        """
        if '/entry/instrument/NDAttributes/CT-Kamera-Z' in self.f:
            self.distance_proxy = self.f['/entry/instrument/NDAttributes/CT-Kamera-Z']
            self.doubleSpinBox_distance_2.setValue(round(self.distance_proxy[-1] + 25))
            #print('Function prefill_distance:', round(self.distance_proxy[-1] + 25))
        else:
            self.doubleSpinBox_distance_2.setValue(0)
            #print('Function prefill_distance: Not found. Set to 0')
        """
        QtWidgets.QApplication.processEvents()


    def load_data(self):
        self.f = h5py.File(self.path_klick, 'r', libver='latest', swmr=True)
        self.w = self.graph

        #prefill rotation-speed[°/img]        #Polynom fit for the angles
        poly_coeff = numpy.polyfit(numpy.arange(len(self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ])), self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ], 1, rcond=None, full=False, w=None, cov=False)
        self.speed_W = poly_coeff[0]
        self.number_of_used_projections = round(180 / self.speed_W)

        # load recent 180deg sino
        self.vol_proxy = self.f['/entry/data/data']
        Sino = self.vol_proxy[- self.number_of_used_projections : , self.slice_number.value(), :]
        self.Norm = Sino


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

        if self.grid_micrometer.isChecked() == True:
            print('ADDING RULER MICROMETER')

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


        if self.grid_pixel.isChecked() == True:
            print('ADDING PXL RULER')

            #add label x
            cv2.putText(self.slice, 'Pixel', (
                round(self.slice.shape[1] / 2) + 20,
                round(self.ruler_grid_line_thickness) * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                        thickness=self.ruler_grid_line_thickness)

            #add ruler +X

            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                print('drawing a line from ', r, 0, 'to ', r, self.slice.shape[1])
                cv2.line(self.slice, (r,0),   (r,self.slice.shape[1]), self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

            # add ruler -X
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                #print(r)
                cv2.line(self.slice, (r, 0),   (r, self.slice.shape[1]), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)


            # add label Y
            cv2.putText(self.slice, 'Pixel',
                        (20, round(self.slice.shape[1] / 2) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                        thickness=self.ruler_grid_line_thickness)

            # add ruler +Y
            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

            # add ruler -Y
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)
            print('DONE ADDING RULER')
        return self.slice




    def reconstruct(self):
        QtWidgets.QApplication.processEvents()

        self.full_size = self.Norm.shape[1]
        self.number_of_projections = self.Norm.shape[0]

        self.number_of_used_projections = round(180 / self.speed_W)

        # create list with all projection angles
        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W + self.graph[-1] + self.rotation_offset) * math.pi / 180



        if self.pixel_size.value() != 1:
            if self.pixel_proxy[-1] == 3.61:
                self.COR = self.COR_1.value()
            elif self.pixel_proxy[-1] == 1.44:
                self.COR = self.COR_2.value()
            elif self.pixel_proxy[-1] == 0.72:
                self.COR = self.COR_3.value()
            elif self.pixel_proxy[-1] == 0.36:
                self.COR = self.COR_4.value()
            else:
                self.COR = 50
        else:
            self.COR = self.COR_1.value()
            print('Pixel Size unknown. Use COR_1')

        center_list = [self.COR + round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size)] * (self.number_of_used_projections)

        # create one sinogram in the form [z, y, x]
        transposed_sinos = numpy.zeros((self.number_of_used_projections, 1, self.full_size), dtype=float)
        transposed_sinos[:,0,:] = self.Norm[-self.number_of_used_projections : , : ]

        #extend data with calculated parameter, compute logarithm, remove NaN-values
        log_sinos = tomopy.minus_log(transposed_sinos)
        log_sinos = numpy.nan_to_num(log_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
        extended_sinos = tomopy.misc.morph.pad(log_sinos, axis=2,
                                               npad=round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size),
                                               mode='edge')

        #reconstruct one slice

        if self.GPU_CUDA.isChecked() == True:
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra, options=options)
        else:
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm='gridrec', filter_name='shepp')


        slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm='gridrec', filter_name='shepp')
        slices = slices[:,round(self.full_size/4):-round(self.full_size/4),round(self.full_size/4):-round(self.full_size/4)]
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0,val=10)
        slices = slices * (10000 / self.pixel_size.value())
        self.slice = slices[0,:,:]   #reduce dimensions from 3 to 2

        self.add_ruler()

        # set image dimensions only for the first time or when scan-type was changed
        if self.new == 1:
            self.reco_rec['dimension'] = [
                {'size': self.slice.shape[1], 'fullSize': self.slice.shape[1], 'binning': 1},
                {'size': self.slice.shape[0], 'fullSize': self.slice.shape[0], 'binning': 1}]
            self.new = 0

        # write reconstruction result to the reconstruction PV
        self.reco_rec['value'] = (
            {'floatValue': self.slice.flatten().astype(numpy.float32)},
        )


#=======================================================================================================================
#no idea why we need this, but it wouldn't work without it ;-)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = OnTheFlyNavigator()
    main.show()
    sys.exit(app.exec_())

#end of code
