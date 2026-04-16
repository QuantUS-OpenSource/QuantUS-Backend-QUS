import math
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import firwin, lfilter, hilbert
from scipy.ndimage import correlate

from .objects import DataOutputStruct, InfoStruct, ScParams
from ..transforms import scanConvert3dVolumeSeries
from .philipsRf import Rfdata, parseRF

class InfoStruct3d(InfoStruct):
    def __init__(self):
        super().__init__()
        self.zResRF: float
        self.coronalRes: float
        self.axialLen: float
        self.lateralLen: float
        self.coronalLen: float
        self.frameRate: float
        self.scParams: ScParams

def QbpFilter(rfData: np.ndarray, Fc1: float, Fc2: float, FiltOrd: int) -> Tuple[np.ndarray, np.ndarray]:
    FiltCoef = firwin(FiltOrd+1, [Fc1*2, Fc2*2], window="hamming", pass_zero="bandpass") # type: ignore
    FiltRfDat = np.transpose(lfilter(np.transpose(FiltCoef),1,np.transpose(rfData)))

    # Do Hilbert Transform on each column
    IqDat = np.zeros(FiltRfDat.shape).astype(np.complex128)
    DbEnvDat = np.zeros(FiltRfDat.shape)
    for i in range(FiltRfDat.shape[1]):
        IqDat[:,i] = hilbert(FiltRfDat[:,i])
        DbEnvDat[:,i] = 20*np.log10(abs(IqDat[:,i])+1)
    
    return IqDat, DbEnvDat

def bandpassFilterEnvLog(rfData: np.ndarray, scParams: ScParams) -> Tuple[np.ndarray, np.ndarray]:
    # Below params are from Philips trial & error
    QbpFiltOrd = 80
    QbpFcA1 = 0.026
    QbpFcA2 = 0.068
    QbpFcB1 = 0.030
    QbpFcB2 = 0.072
    QbpFcC1 = 0.020
    QbpFcC2 = 0.064

    R, M, C = rfData.shape
    rfDat2 = rfData.reshape(R, -1, order='F')
    IqDatA, DbEnvDatA = QbpFilter(rfDat2, QbpFcA1, QbpFcA2, QbpFiltOrd)
    IqDatB, DbEnvDatB = QbpFilter(rfDat2, QbpFcB1, QbpFcB2, QbpFiltOrd)
    IqDatC, DbEnvDatC = QbpFilter(rfDat2, QbpFcC1, QbpFcC2, QbpFiltOrd)
    DbEnvDat = (DbEnvDatA + DbEnvDatB + DbEnvDatC)/3
    QbpDecimFct = int(np.ceil(DbEnvDat.shape[0])/2048)
    DbEnvDat = correlate(DbEnvDat, np.ones((QbpDecimFct,1))/QbpDecimFct, mode='nearest')
    DbEnvDat = DbEnvDat[np.arange(0, DbEnvDat.shape[0],QbpDecimFct)]
    NumSamples = DbEnvDat.shape[0]
    NumPlanes = scParams.NUM_PLANES

    # Format RF data to match B-MODE (DbEnvDat)
    formattedRf = rfDat2[np.arange(0, DbEnvDatA.shape[0],QbpDecimFct)]
    rfFullVol = formattedRf[:,:scParams.NumRcvCols*NumPlanes].reshape(NumSamples,scParams.NumRcvCols,NumPlanes, order='F')

    # Keep first full volume
    DbEnvDat_FullVol = DbEnvDat[:,:scParams.NumRcvCols*NumPlanes].reshape(NumSamples,scParams.NumRcvCols,NumPlanes, order='F')
    return DbEnvDat_FullVol, rfFullVol


def sort3DData(dataIn: Rfdata, scParams: ScParams) -> Tuple[np.ndarray, ScParams]:
    dataOut = dataIn.echoData[0]

    # Compute the number of columns and receive beams for use later
    OutML_Azim = dataIn.dbParams.azimuthMultilineFactorXbrOut[0]
    scParams.NumXmtCols = int(max(dataIn.headerInfo.Line_Index))+1
    scParams.NumRcvCols = int(OutML_Azim*scParams.NumXmtCols)
    
    return dataOut, scParams

def readSIPscVDBParams(filename):
    print("Reading SIP scan conversion VDB Params...")
    file = open(filename, "r")
    scParams = ScParams()
    for line in file:
        paramName, paramValue = line.split(" = ")
        try: 
            paramValue, _ = paramValue.split(" \n")
        except ValueError:
            paramValue, _ = paramValue.split(" ,")
        paramAr = paramValue.split(" ")
        for i in range(len(paramAr)):
            paramAr[i] = float(paramAr[i]) # type: ignore

        if len(paramAr) == 1:
            paramValue = paramAr[0]
        else:
            paramValue = paramAr

        if (paramName == 'VDB_2D_ECHO_MATRIX_ELEVATION_NUM_TRANSMIT_PLANES'):
            scParams.NUM_PLANES = int(paramValue) # type: ignore
        elif (paramName == 'pixPerMm'):
            scParams.pixPerMm = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_APEX_TO_SKINLINE'):
            scParams.VDB_2D_ECHO_APEX_TO_SKINLINE = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_START_WIDTH_GC'):
            scParams.VDB_2D_ECHO_START_WIDTH_GC = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_STOP_WIDTH_GC'):
            scParams.VDB_2D_ECHO_STOP_WIDTH_GC = paramValue # type: ignore
        elif (paramName == 'VDB_THREED_START_ELEVATION_ACTUAL'):
            scParams.VDB_THREED_START_ELEVATION_ACTUAL = paramValue # type: ignore
        elif (paramName == 'VDB_THREED_STOP_ELEVATION_ACTUAL'):
            scParams.VDB_THREED_STOP_ELEVATION_ACTUAL = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_STOP_DEPTH_SIP'):
            scParams.VDB_2D_ECHO_STOP_DEPTH_SIP = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_START_DEPTH_SIP'):
            scParams.VDB_2D_ECHO_START_DEPTH_SIP = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_SLACK_TIME_MM'):
            scParams.VDB_2D_ECHO_SLACK_TIME_MM = paramValue # type: ignore
        elif (paramName == 'VDB_THREED_RT_VOLUME_RATE'):
            scParams.VDB_THREED_RT_VOLUME_RATE = paramValue # type: ignore

    file.close()        
    print('Finished reading SIP scan converstion VDB params...')
    return scParams

def getVolume(rfPath: str, sipNumOutBits: int = 8, DRlowerdB: int = 20, DRupperdB: int = 40):
    rfPath = Path(rfPath)
    scParamFname = f"{rfPath.name[:-3]}_Extras.txt"
    scParamPath = rfPath.parent / Path(scParamFname)

    # #Read in parameter data (primarily for scan conversion)
    scParams = readSIPscVDBParams(scParamPath)
    scParams.pixPerMm=2.5; #for scan conversion grid
    # TODO: implement handling for IQ data (see scParams.removeGapsFlag in Dave Duncan MATLAB code)

    #Read in the interleaved SIP volume data time series (both linear/non-linear parts) 
    rawData = parseRF(f"{rfPath.absolute()}", 0, 2000)

    rfDataArr, scParams = sort3DData(rawData, scParams)

    #Bandpass Filtering + Envelope Det + Log Compression
    dBEnvData_vol, rfVol = bandpassFilterEnvLog(rfDataArr,scParams)

    #Scan Conversion of 3D volume time series (Only doing 1 volume here)
    SC_Vol, SC_coord_map, bmodePhysicalDims = scanConvert3dVolumeSeries(dBEnvData_vol, scParams, scale=False)
    # SC_rfVol, rfDims = scanConvert3dVolumeSeries(rfVol, scParams, normalize=False)

    #Parameters for basic visualization of volume
    slope = (2**sipNumOutBits)/(20*np.log10(2**sipNumOutBits))
    upperLim = slope * DRupperdB
    lowerLim = slope * DRlowerdB

    # Format image for output
    SC_Vol = np.clip(SC_Vol, lowerLim, upperLim)
    SC_Vol = (SC_Vol - lowerLim)/(upperLim - lowerLim) * 255
    preSC_Vol = np.clip(dBEnvData_vol, lowerLim, upperLim)
    preSC_Vol = (preSC_Vol - lowerLim)/(upperLim - lowerLim) * 255
    # bmodePhysicalDims = [bmodePhysicalDims[2], bmodePhysicalDims[0], bmodePhysicalDims[1]]
    # rfDims = [rfDims[2], rfDims[0], rfDims[1]]

    Data = DataOutputStruct()
    Data.rf = rfVol
    Data.bMode = dBEnvData_vol
    Data.scBmode = SC_Vol
    Data.widthPixels = SC_Vol.shape[2]
    Data.depthPixels = SC_Vol.shape[1]
    Data.coordMap3d = SC_coord_map

    Info = InfoStruct3d()
    Info.minFrequency = 1000000
    Info.maxFrequency = 6000000
    Info.lowBandFreq = 1000000
    Info.upBandFreq = 6000000
    Info.centerFrequency = 3000000 #Hz
    Info.samplingFrequency = 50000000 # TODO: currently a guess
    # bmodePhysicalDims = [Lateral_mm, Elevation_mm, Axial_mm]
    # dBEnvData_vol shape = (depth/axial, lateral, elevation)
    # SC_Vol shape = (Axial, Lateral, Elevation)
    Info.axialLen = bmodePhysicalDims[2]
    Info.lateralLen = bmodePhysicalDims[0]
    Info.coronalLen = bmodePhysicalDims[1]
    Info.xResRF = bmodePhysicalDims[0] / dBEnvData_vol.shape[0] # mm/pixel (axial)
    Info.yResRF = bmodePhysicalDims[1] / dBEnvData_vol.shape[1] # mm/pixel (lateral)
    Info.zResRF = bmodePhysicalDims[2] / dBEnvData_vol.shape[2] # mm/pixel (coronal/elevation)
    Info.axialRes   = bmodePhysicalDims[0] / SC_Vol.shape[0] # mm/pixel
    Info.lateralRes = bmodePhysicalDims[1] / SC_Vol.shape[1] # mm/pixel
    Info.coronalRes = bmodePhysicalDims[2] / SC_Vol.shape[2] # mm/pixel
    Info.scParams = scParams
    
    return Data, Info

def philipsRfParser3d(scanPath: Path, phantomPath: Path) \
    -> Tuple[DataOutputStruct, InfoStruct3d, DataOutputStruct, InfoStruct3d]:
    """Parses Philips 3D RF data and metadata.
    
    Args:
        filePath (str): Path to the RF data.
        phantomPath (str): Path to the phantom data.
    
    Returns:
        Tuple: RF data and metadata for image and phantom.
    """
    imgData, imgInfo = getVolume(scanPath)
    pkl_path = str(scanPath).replace('.rf', '.pkl')
    with open(pkl_path, 'wb') as f:
        import pickle
        pickle.dump([imgData, imgInfo], f)
    
    phantomData, phantomInfo = getVolume(phantomPath)
    pkl_path = str(phantomPath).replace('.rf', '.pkl')
    with open(pkl_path, 'wb') as f:
        import pickle
        pickle.dump([phantomData, phantomInfo], f)
    return imgData, imgInfo, phantomData, phantomInfo