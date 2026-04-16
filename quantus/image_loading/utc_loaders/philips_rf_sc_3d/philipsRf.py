import os
import platform
from pathlib import Path
from datetime import datetime
import warnings
import ctypes as ct

import numpy as np
from scipy.io import savemat
from philipsRfParser import getPartA, getPartB


class HeaderInfoStruct:
    """Philips-specific structure containing information from the headers."""
    def __init__(self):
        self.RF_CaptureVersion: list
        self.Tap_Point: list
        self.Data_Gate: list
        self.Multilines_Capture: list
        self.Steer: list
        self.elevationPlaneOffset: list
        self.PM_Index: list
        self.Pulse_Index: list
        self.Data_Format: list
        self.Data_Type: list
        self.Header_Tag: list
        self.Threed_Pos: list
        self.Mode_Info: list
        self.Frame_ID: list
        self.CSID: list
        self.Line_Index: list
        self.Line_Type: list
        self.Time_Stamp: int
        self.RF_Sample_Rate: list

class dbParams:
    """Philips-specific structure containing signal properties of the scan."""
    def __init__(self):
        self.acqNumActiveScChannels2d: list
        self.azimuthMultilineFactorXbrOut: list
        self.azimuthMultilineFactorXbrIn: list
        self.numOfSonoCTAngles2dActual: list
        self.elevationMultilineFactor: list
        self.numPiPulses: list
        self.num2DCols: list
        self.fastPiEnabled: list
        self.numZones2d: list
        self.numSubVols: list
        self.numPlanes: list
        self.zigZagEnabled: list
        self.azimuthMultilineFactorXbrOutCf: list
        self.azimuthMultilineFactorXbrInCf: list
        self.multiLineFactorCf: list
        self.linesPerEnsCf: list
        self.ensPerSeqCf: list
        self.numCfCols: list
        self.numCfEntries: list
        self.numCfDummies: list
        self.elevationMultilineFactorCf: list
        self.Planes: list
        self.tapPoint: list

class Rfdata:
    """Philips-specific structure containing donstructed RF data."""
    def __init__(self):
        self.lineData: np.ndarray # Array containing interleaved line data (Data x XmitEvents)
        self.lineHeader: np.ndarray # Array containing qualifier bits of the interleaved line data (Qualifiers x XmitEvents)
        self.headerInfo = HeaderInfoStruct() # Structure containing information from the headers
        self.echoData: np.ndarray # Array containing echo line data
        self.dbParams = dbParams() # Structure containing dbParameters. Should match feclib::RFCaptureDBInfo
        self.echoMModeData: list
        self.miscData: list

def findSignature(filepath: Path):
    file = open(filepath, 'rb')
    sig = list(file.read(8))
    return sig

def callGetPartA(numClumps: int, filename: str, offset: int) -> np.ndarray:
    partA = getPartA(numClumps, filename, offset)
    partA = np.array(partA, dtype=int).reshape((12, numClumps), order='F')
    return partA

def callGetPartB(numClumps: int, filename: str, offset: int) -> np.ndarray:
    partB = getPartB(numClumps, filename, offset)
    partB = np.array([partB], dtype=int)
    return partB

def pruneData(lineData, lineHeader, ML_Capture):
    # Remove false gate data at beginning of the line
    numSamples = lineData.shape[0]
    referenceLine = int(np.ceil(lineData.shape[1]*0.2))-1    
    startPoint = int(np.ceil(numSamples*0.015))-1
    indicesFound = np.where(lineHeader[startPoint:numSamples+1, referenceLine]==3)
    if not len(indicesFound[0]):
        iFirstSample = 1
    else:
        iFirstSample = indicesFound[0][0]+startPoint
    
    # Align the pruning
    alignment = np.arange(0,numSamples, np.double(ML_Capture))
    diff = alignment - iFirstSample
    iFirstSample = int(alignment[np.where(diff>=0)[0][0]])
    
    # Prune data
    prunedData = lineData[iFirstSample:numSamples+1,:]
    lineHeader = lineHeader[iFirstSample:numSamples+1,:]
               
    # Remove zero data at end of the line
    # Start from last 1 of the line
    numSamples = prunedData.shape[0]
    startPoint = int(np.floor(numSamples*0.99))-1
    indicesFound = np.where(lineHeader[startPoint:numSamples+1,referenceLine]==0)
    if not len(indicesFound[0]):
        iLastSample = numSamples
    else:
        iLastSample = indicesFound[0][0]+startPoint
        # Align the pruning
        alignment = np.arange(0,numSamples, np.double(ML_Capture))
        diff = alignment - iLastSample
        iLastSample = int(alignment[np.where(diff >= 0)[0][0]])-1
    
    # Prune data
    prunedData = prunedData[:iLastSample+1, :]

    return prunedData

def SortRF(RFinput, Stride, ML, CRE=1, isVoyager=True):
    # Initialize default parameters
    N = RFinput.shape[0]
    xmitEvents = RFinput.shape[1]
    depth = int(np.floor(N/Stride))
    MLs = np.arange(0,ML)

    # Make into Column Vecor
    MLs = MLs[:]

    out1 = np.array([])
    out2 = np.array([])
    out3 = np.array([])
    
    # Preallocate output array, but only for those that will be used
    if CRE == 4:
        out3 = np.zeros((depth, ML, xmitEvents))
        out2 = np.zeros((depth, ML, xmitEvents))
        out1 = np.zeros((depth, ML, xmitEvents))
        out0 = np.zeros((depth, ML, xmitEvents))
    elif CRE == 3:
        out2 = np.zeros((depth, ML, xmitEvents))
        out1 = np.zeros((depth, ML, xmitEvents))
        out0 = np.zeros((depth, ML, xmitEvents))
    elif CRE == 2:
        out1 = np.zeros((depth, ML, xmitEvents))
        out0 = np.zeros((depth, ML, xmitEvents))
    elif CRE == 1:
        out0 = np.zeros((depth, ML, xmitEvents))
    
    if ((CRE != 1) and (CRE != 2) and (CRE != 4)):
        print("\nno sort list for this CRE\n")
    
    if Stride == 128:
        ML_SortList = list(range(128))
    elif Stride == 32:
        if CRE == 4:
            ML_SortList = [4, 4, 5, 5, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
        else:
            ML_SortList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif Stride == 16:
        if CRE == 1:
            ML_SortList = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
        elif CRE == 2:
            ML_SortList = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
        elif CRE == 4:
            ML_SortList = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
    elif Stride == 12:
        if CRE ==1:
            ML_SortList = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
        elif CRE == 2:
            ML_SortList = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        elif CRE == 4:
            ML_SortList = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
    elif Stride == 8:
        if CRE == 1:
            ML_SortList = [0, 2, 4, 6, 1, 3, 5, 7]
        elif CRE == 2:
            ML_SortList = [0, 1, 2, 3, 0, 1, 2, 3]
        elif CRE == 4:
            ML_SortList = [0, 0, 1, 1, 0, 0, 1, 1]
    elif Stride == 4:
        if CRE == 1:
            ML_SortList = [0, 2, 1, 3]
        elif CRE == 2:
            ML_SortList = [0, 1, 0, 1]
        elif CRE == 4:
            ML_SortList = [0, 0, 0, 0]
    elif Stride == 2:
        if CRE == 1:
            ML_SortList = [0, 1]
        elif CRE == 2:
            ML_SortList = [0, 0]
        elif CRE == 4:
            ML_SortList = [0, 0]
    else:
        print ("\nno sort list for this stride\n")
    
    if ((ML-1)>max(ML_SortList)) or (CRE == 4 and Stride < 16) or (CRE == 2 and Stride < 4):
        print ("\nCaptured ML is insufficient, some ML were not captured\n")
    
    # Sort 
    for k in range(ML):
        iML = np.where(ML_SortList == MLs[k])[0]
        out0[:depth, k, :] = RFinput[np.arange(iML[0],(depth*Stride), Stride)]
        if CRE == 2:
            out1[:depth, k, :] = RFinput[np.arange(iML[1], (depth*Stride), Stride), :]
            out2[:depth,k,:] = RFinput[np.arange(iML[1], (depth*Stride), Stride), :]
            out3[:depth,k,:] = RFinput[np.arange(iML[1], (depth*Stride), Stride), :]
        elif CRE == 4:
            out2[:depth, k, :] = RFinput[np.arange(iML[2], (depth*Stride), Stride), :]
            out3[:depth, k, :] = RFinput[np.arange(iML[3], (depth*Stride), Stride), :]

    return out0, out1, out2, out3

     

def parseDataF(rawrfdata, headerInfo):
    # Definitions
    minNeg = 2**18 # Used to convert integers to 2's complement

    # Find header clumps
    # iHeader pts to the index of the header clump
    # Note that each Header is exactly 1 "Clump" long
    iHeader = np.array(np.where((rawrfdata[0,:] & 1572864)==524288))[0]
    numHeaders = len(iHeader) - 1 # Ignore last header as it is a part of a partial line

    # Get maximum number of samples between consecutive headers
    maxNumSamples = 0
    for m in range(numHeaders):
        tempMax = iHeader[m+1] - iHeader[m] - 1
        if (tempMax > maxNumSamples):
            maxNumSamples = tempMax
    
    numSamples = maxNumSamples*12

    # Preallocate arrays
    lineData = np.zeros((numSamples, numHeaders), dtype = np.int32)
    lineHeader = np.zeros((numSamples, numHeaders), dtype = np.uint8)

    # Extract data
    for m in range(numHeaders):
        iStartData = iHeader[m]+2
        iStopData = iHeader[m+1]-1

        if headerInfo.Data_Type[m] == float(0x5a):
            # set stop data to a reasonable value to keep file size form blowing up
            iStopData = iStartData + 10000
        
        # Get Data for current line and convert to 2's complement values
        lineData_u32 = rawrfdata[:12,iStartData:iStopData+1]
        lineData_s32 = np.int32(lineData_u32&524287)
        iNeg = np.where(lineData_s32 >= minNeg)
        lineData_s32[iNeg] -= (2*minNeg) # type: ignore
        # for point in iNeg:
        #     lineData_s32[point[0], point[1]] = lineData_s32[point[0], point[1]]-2*minNeg
        lineHeader_u8 = (lineData_u32 & 1572864) >> 19

        # lineData[0:lineData_s32.size,m] = lineData_s32[:lineData_s32.size]

        lineData[:lineData_s32.size,m] = lineData_s32.ravel(order='F')
        lineHeader[:lineHeader_u8.size,m] = lineHeader_u8.ravel(order='F')

    return lineData, lineHeader


def parseHeaderV(rawrfdata):
    temp_headerInfo = HeaderInfoStruct()

    iHeader = np.where(np.uint8(rawrfdata[2,0,:])&224)
    numHeaders = len(iHeader)-1 # Ignore last header as it is part of a partial line

    # Get infor for each header
    for m in range(numHeaders):
        packedHeader = ''
        for k in np.arange(11,0,-1):
            temp = ''
            for i in np.arange(2,0,-1):
                temp += bin(np.uint8(rawrfdata[i,k,iHeader[m]]))

            # Discard first 3 bits, redundant info
            packedHeader += temp[3:24]

               

        iBit = 0
        temp_headerInfo.RF_CaptureVersion[m] = int(packedHeader[iBit:iBit+4],2)
        iBit += 4
        temp_headerInfo.Tap_Point[m] = int(packedHeader[iBit:iBit+3],2)
        iBit += 3
        temp_headerInfo.Data_Gate[m] = int(packedHeader[iBit],2)
        iBit += 1
        temp_headerInfo.Multilines_Capture[m] = int(packedHeader[iBit:iBit+4],2)
        iBit += 4
        temp_headerInfo.RF_Sample_Rate[m] = int(packedHeader[iBit],2)
        iBit += 1
        temp_headerInfo.Steer[m] = int(packedHeader[iBit:iBit+6],2)
        iBit += 6
        temp_headerInfo.elevationPlaneOffset[m] = int(packedHeader[iBit:iBit+8],2)
        iBit += 8
        temp_headerInfo.PM_Index[m] = int(packedHeader[iBit:iBit+2],2)
        iBit += 2
        temp_headerInfo.Line_Index[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Pulse_Index[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Data_Format[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Data_Type[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Header_Tag[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Threed_Pos[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Mode_Info[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Frame_ID[m] = int(packedHeader[iBit:iBit+32],2)
        iBit += 32
        temp_headerInfo.CSID[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Line_Type[m] = int(packedHeader[iBit:iBit+16],2)
        iBit += 16
        temp_headerInfo.Time_Stamp = int(packedHeader[iBit:iBit+32],2)

    return temp_headerInfo

def getFillerZeros(num):
    zeros = "0"
    num -= 1
    while num > 0:
        zeros += "0"
        num -= 1
    return zeros

def parseHeaderF(rawrfdata):

    # Find header clumps
    # iHeader pts to the index of the header clump
    # Note that each header is exactly 1 "Clump" long
    iHeader = np.array(np.where(rawrfdata[0,:]&1572864 == 524288))[0]
    numHeaders: int = iHeader.size - 1 # Ignore last header as it is a part of a partial line

    HeaderInfo = HeaderInfoStruct()

    HeaderInfo.RF_CaptureVersion = np.zeros(numHeaders, dtype=np.uint8)
    HeaderInfo.Tap_Point = np.zeros(numHeaders, np.uint8)
    HeaderInfo.Data_Gate = np.zeros(numHeaders, np.uint8)
    HeaderInfo.Multilines_Capture = np.zeros(numHeaders, np.uint8)
    HeaderInfo.RF_Sample_Rate = np.zeros(numHeaders, np.uint8)
    HeaderInfo.Steer = np.zeros(numHeaders, np.uint8)
    HeaderInfo.elevationPlaneOffset = np.zeros(numHeaders, np.uint8)
    HeaderInfo.PM_Index = np.zeros(numHeaders, np.uint8)
    HeaderInfo.Line_Index = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Pulse_Index = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Data_Format = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Data_Type = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Header_Tag = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Threed_Pos = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Mode_Info = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Frame_ID = np.zeros(numHeaders, np.uint32)
    HeaderInfo.CSID = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Line_Type = np.zeros(numHeaders, np.uint16)
    HeaderInfo.Time_Stamp = np.zeros(numHeaders, np.uint32)


    # Get info for Each Header
    for m in range(numHeaders):
        packedHeader = bin(rawrfdata[12, iHeader[m]])[2:]
        remainingZeros = 4 - len(packedHeader)
        if remainingZeros > 0:
            zeros = getFillerZeros(remainingZeros)
            packedHeader = str(zeros + packedHeader)
        for i in np.arange(11,-1,-1):
            curBin = bin(int(rawrfdata[i,iHeader[m]]))[2:]
            remainingZeros = 21 - len(curBin)
            if remainingZeros > 0:
                zeros = getFillerZeros(remainingZeros)
                curBin = str(zeros + curBin)
            packedHeader += curBin
                

        iBit = 2
        HeaderInfo.RF_CaptureVersion[m] = int(packedHeader[iBit:iBit+4], 2)
        iBit += 4
        HeaderInfo.Tap_Point[m] = int(packedHeader[iBit:iBit+3], 2)
        iBit += 3
        HeaderInfo.Data_Gate[m] = int(packedHeader[iBit], 2)
        iBit += 1
        HeaderInfo.Multilines_Capture[m] = int(packedHeader[iBit:iBit+4], 2)
        iBit += 4
        iBit += 15 # Waste 15 bits (unused)
        HeaderInfo.RF_Sample_Rate[m] = int(packedHeader[iBit], 2)
        iBit += 1
        HeaderInfo.Steer[m] = int(packedHeader[iBit:iBit+6], 2)
        iBit += 6
        HeaderInfo.elevationPlaneOffset[m] = int(packedHeader[iBit:iBit+8], 2)
        iBit += 8
        HeaderInfo.PM_Index[m] = int(packedHeader[iBit:iBit+2], 2)
        iBit += 2
        HeaderInfo.Line_Index[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Pulse_Index[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Data_Format[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Data_Type[m] = int(packedHeader[iBit: iBit+16], 2)
        iBit += 16
        HeaderInfo.Header_Tag[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Threed_Pos[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Mode_Info[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Frame_ID[m] = int(packedHeader[iBit:iBit+32], 2)
        iBit += 32
        HeaderInfo.CSID[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Line_Type[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Time_Stamp[m] = int(str(packedHeader[iBit:iBit+13]+packedHeader[iBit+15:iBit+34]), 2)
    
    return HeaderInfo




def parseDataV(rawrfdata, headerInfo):
    # Definitions
    minNeg = 16*(2^16) # Used to convert offset integers to 2's complement

    # Find header clumps
    # iHeader pts to the index of the header clump
    # Note that each Header is exactly 1 "Clump" long
    iHeader = np.where(rawrfdata[2,0,:]&224==64)
    numHeaders = len(iHeader)-1 # Ignore last header as it is a part of a partial line
    numSamples = (iHeader[1]-iHeader[0]-1)*12
    
    # Preallocate arrays
    lineData = np.zeros((numSamples, numHeaders), dtype = np.int32)
    lineHeader = np.zeros((numSamples, numHeaders), dtype = np.uint8)

    # Extract data
    for m in range(len(numHeaders)):

        # Get data in between headers
        iStartData = iHeader[m]+1
        iStopData = iHeader[m+1]-1

        # Push pulses (DT 0x5a) are very long, and have no valid RX data
        if headerInfo.Data_Type[m] == float(0x5a):
            # set stop data to a reasonable value to keep file size from blowing up
            iStopData = iStartData+10000
        
        # Get Data for current line and convert to 2's complement values
        lineData_u8 = rawrfdata[:,:,iStartData:iStopData]
        lineData_s32 = np.int32(lineData_u8[0,:,:])+np.int32(lineData_u8[1,:,:])*2^8+np.int32(lineData_u8[2,:,:]&np.uint8(31))*2^16
        iNeg = np.where(lineData_s32>=minNeg)
        lineData_s32[iNeg] = lineData_s32[iNeg]-2*minNeg
        lineHeader_u8 = (lineData_u8[2,:,:]&224)>>6

        lineData[:lineData_s32.size-1,m] = lineData_s32[:lineData_s32.size-1]
        lineHeader[:lineHeader_u8.size-1,m] = lineHeader_u8[:lineHeader_u8.size-1]

    return lineData, lineHeader


def parseFileHeader(file_obj, endianness):
    fileVersion = int.from_bytes(file_obj.read(4), endianness, signed=False)
    numFileHeaderBytes = int.from_bytes(file_obj.read(4), endianness, signed=False)
    print("\tFile Version: {0}\n\tHeader Size: {1} bytes\n".format(fileVersion, numFileHeaderBytes))

    # Handle accordingly to fileVersion
    temp_dbParams = dbParams()
    if fileVersion == 2:
        temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)], (14, 11), order='F')
        temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
        temp_dbParams.numPlanes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
        temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

    elif fileVersion == 3:
        temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
        temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numSubVols = int.from_bytes(file_obj.read(4), endianness, signed=False)
        temp_dbParams.numPlanes = int.from_bytes(file_obj.read(4), endianness, signed=False)
        temp_dbParams.zigZagEnabled = int.from_bytes(file_obj.read(4), endianness, signed=False)

        temp_dbParams.multiLineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
        temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
        temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]

    elif fileVersion == 4:
        temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.azimuthMultilineFactorXbrOutCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrInCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.elevationMultilineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
        temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

        temp_dbParams.Planes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

        temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

        temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
        temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

    elif fileVersion == 5:
        temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.azimuthMultilineFactorXbrOutCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrInCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.elevationMultilineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.multiLineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
        temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

        temp_dbParams.numPlanes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

        temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

        temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
        temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

    elif fileVersion == 6:
        temp_dbParams.tapPoint = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
        temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrOutCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.azimuthMultilineFactorXbrInCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.elevationMultilineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.multiLineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
        temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
        temp_dbParams.numPlanes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
        temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
        temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
        temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
        temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

    else:
        numFileHeaderBytes = 0
        print("\nUnknown file version\n")

    return temp_dbParams, numFileHeaderBytes


def parseRF(filepath: str, readOffset: int, readSize: int) -> Rfdata:
    """Open and parse RF data file"""
    # Remember to make sure .c files have been compiled before running

    rfdata = Rfdata()
    print (str ("Opening: " + filepath))
    file_obj = open(filepath, 'rb')

    # Voyager or Fusion?
    VHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 160, 160]
    FHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 11, 11 ]
    fileHeaderSize = len(VHeader)
    
    fileHeader = list(file_obj.read(fileHeaderSize))
    isVoyager = False
    hasFileHeader = False

    if fileHeader == VHeader:
        print("Header information found ...")
        print("Parsing Voyager RF capture file ...")
        isVoyager = True
        hasFileHeader = True
    elif fileHeader == FHeader:
        print("Header information found:")
        print("Parsing Fusion RF capture file ...")
        hasFileHeader = True
    else: # Legacy V-ACB file
        print("Parsing Voyager RF capture file ...")
        isVoyager = True


    # Load RAW RF data
    start_time = datetime.now()

    # Read out file header info
    endianness = 'little'
    if hasFileHeader:
        if isVoyager:
            endianness = 'big'      

        [rfdata.dbParams, numFileHeaderBytes] = parseFileHeader(file_obj, endianness)
        totalHeaderSize = fileHeaderSize+8+numFileHeaderBytes # 8 bytes from fileVersion and numFileHeaderBytes
        # fseek(fid, totalHeaderSize, 'bof')
    else:
        # rfdata.dbParams = []
        totalHeaderSize = 0
    
    readOffset *= (2**20)
    remainingSize = os.stat(filepath).st_size - totalHeaderSize
    readSize *= (2**20)

    if isVoyager:
        # Align read offset and size
        alignment = np.arange(0,remainingSize+1,36)
        offsetDiff = alignment - readOffset
        readDiff = alignment - readSize
        readOffset = alignment[np.where(offsetDiff >= 0)[0][0]].__int__()
        readSize = alignment[np.where(readDiff >= 0)[0][0]].__int__()
        
        # Start reading
        rawrfdata = open(filepath,'rb').read(readSize.__int__())
    
    else: # isFusion
        # Align read and offset size
        alignment = np.arange(0,remainingSize+1,32)
        offsetDiff = alignment - readOffset
        readDiff = alignment - readSize 

        matchingIndices = np.where(offsetDiff >= 0)[0]
        if len(matchingIndices) > 0:
            readOffset = alignment[matchingIndices[0]].__int__()
        else:
            readOffset = 0

        matchingIndices = np.where(readDiff >= 0)[0]
        if len(matchingIndices) > 0:
            readSize = alignment[matchingIndices[0]].__int__()
        else:
            readSize = remainingSize
        numClumps = int(np.floor(readSize/32)) # 256 bit clumps

        offset = totalHeaderSize+readOffset
        partA = callGetPartA(numClumps, filepath, offset)
        partB = callGetPartB(numClumps, filepath, offset)
        rawrfdata = np.concatenate((partA, partB))

    # Reshape Raw RF Dawta
    if isVoyager:
        numClumps = np.floor(len(rawrfdata)/36) # 1 Clump = 12 Samples (1 Sample = 3 bytes)

        rlimit = 180000000 # Limit ~172 MB for reshape workload, otherwise large memory usage
        if len(rawrfdata)>rlimit:
            numChunks = np.floor(len(rawrfdata/rlimit))
            numremBytes = np.mod(len(rawrfdata),rlimit)
            numClumpGroup = rlimit/36

            temp = np.zeros((numChunks+1,3,12,numClumpGroup))
            m=1
            n=1
            # Reshape array into clumps 
            for i in range(numChunks):
                temp[i]=np.reshape(rawrfdata[m:m+rlimit],(3,12,numClumpGroup))
                m += rlimit
                n += numClumpGroup
            
            # Handle the remaining butes
            if numremBytes > 0:
                temp[numChunks]=np.reshape(rawrfdata[m:numClumps*36+1], (3,12,numClumps-n+1))

            # Combine the reshaped arrays
            rawrfdata = np.concatenate((temp[:]),axis=2)
        
    print(str("Elapsed time is "+str(-1*(start_time - datetime.now()))+" seconds."))

    # Parse Header
    print("Parsing header info ...")
    # Extract header info
    if isVoyager:
        headerInfo = parseHeaderV(rawrfdata)
    else: # isFusion
        # if tapPoint == 7:
        #     headerInfo = parseHeaderAdcF(rawrfdata)
        # else:
        headerInfo = parseHeaderF(rawrfdata)

    print(str("Elapsed time is " + str(-1*(start_time - datetime.now())) + " seconds."))

    # Parse RF Data
    print("Parsing RF data ...")
    # Extract RF datad
    Tap_Point = headerInfo.Tap_Point[0]
    if isVoyager:
        [lineData, lineHeader] = parseDataV(rawrfdata, headerInfo)
    else: # isFusion
        # if Tap_Point == 7: #Post-ADC capture
            # [lineData, lineHeader] = parseDataAdcF(rawrfdata, headerInfo)
        # else:
        [lineData, lineHeader] = parseDataF(rawrfdata, headerInfo)
        Tap_Point = headerInfo.Tap_Point[0]
        if Tap_Point == 0: # Correct for MS 19 bits of 21 real data bits
            lineData = lineData << 2
    
    print (str("Elapsed time is " + str(-1*(start_time - datetime.now())) + " seconds."))

    # Pack data
    rfdata.lineData = lineData
    rfdata.lineHeader = lineHeader
    rfdata.headerInfo = headerInfo

    # Free-up Memory
    del rawrfdata

    # Sort into Data Types
    # De-interleave rfdata
    print("Organizing based on data type ...")

    DataType_ECHO = np.arange(1,15)
    DataType_EchoMMode = 26

    DataType_COLOR = [17, 21, 22, 23, 24]
    DataType_ColorMMode = [27, 28]
    DataType_ColorTDI = 24

    DataType_CW = 16
    DataType_PW = [18,19]

    DataType_Dummy = [20,25,29,30,31]

    DataType_SWI = [90,91]

    # OCI and phantoms
    DataType_Misc = [15,88,89]

    if Tap_Point == 7:
        ML_Capture = 128
    else:
        ML_Capture = np.double(rfdata.headerInfo.Multilines_Capture[0])
    
    if ML_Capture == 0:
        SAMPLE_RATE = np.double(rfdata.headerInfo.RF_Sample_Rate[0])
        if SAMPLE_RATE == 0:
            ML_Capture = 16
        else: # 20MHz Capture
            ML_Capture = 32

    Tap_Point = rfdata.headerInfo.Tap_Point[0]
    if Tap_Point == 7: #Hardware is saving teh tap point as 7 and now we convert it back to 4
        Tap_Point = 4
    namePoint = ['PostShepard', 'PostAGNOS', 'PostXBR', 'PostQBP', 'PostADC']
    print(str("\t"+namePoint[Tap_Point]+"\n\t\tCapture_ML:\t"+str(ML_Capture)+"x\n"))

    xmitEvents = len(rfdata.headerInfo.Data_Type)

    # Find Echo Data
    echo_index = np.zeros(xmitEvents).astype(np.int32)
    for i in range(len(DataType_ECHO)):
        index = ((rfdata.headerInfo.Data_Type & 255) == DataType_ECHO[i]) # Find least significant byte
        echo_index = np.bitwise_or(np.array(echo_index), np.array(index).astype(np.int32))

    if np.sum(echo_index) > 0:
        # Remove false gate data at the beginning of the line
        columnsToDelete =  np.where(echo_index==0)
        pruningLineData = np.delete(rfdata.lineData, columnsToDelete, axis=1)
        pruningLineHeader = np.delete(rfdata.lineHeader, columnsToDelete, axis=1)
        if Tap_Point == 4:
            echoData = pruningLineData
        else:
            echoData = pruneData(pruningLineData, pruningLineHeader, ML_Capture)
        #pre-XBR Sort
        if Tap_Point == 0 or Tap_Point == 1:
            ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrIn[0]*rfdata.dbParams.elevationMultilineFactor[0]
            print(str("\t\tEcho_ML:\t"+str(ML_Actual)+"x\n"))
            CRE = 1
            rfdata.echoData = SortRF(echoData, ML_Capture, ML_Actual, CRE, isVoyager)

        elif Tap_Point == 2: # post-XBR Sort
            ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrOut[0]*rfdata.dbParams.elevationMultilineFactor[0]
            print(str("\t\tEcho_ML:\t"+str(ML_Actual)+"x\n"))
            CRE = rfdata.dbParams.acqNumActiveScChannels2d[0]
            print(str("\t\tCRE:\t"+str(CRE)+"\n"))
            rfdata.echoData = SortRF(echoData, ML_Capture, ML_Actual, CRE, isVoyager)
            
        elif Tap_Point == 4: # post-ADC sort
            ML_Actual = 128
            print(str("\t\tEcho_ML:\t"+str(ML_Actual)+"x\n"))
            CRE = 1
            rfdata.echoData = SortRF(echoData, ML_Actual, ML_Actual, CRE, isVoyager)

        else:
            warnings.warn("Do not know how to sort this data set")

    # Find Echo MMode Data
    echoMMode_index = rfdata.headerInfo.Data_Type == DataType_EchoMMode
    if np.sum(echoMMode_index) > 0:
        echoMModeData = pruneData(rfdata.lineData[:,echoMMode_index], rfdata.lineHeader[:,echoMMode_index], ML_Capture)
        ML_Actual = 1
        print(str("\t\tEchoMMode_ML:\t"+str(ML_Actual)+"x\n"))
        CRE = 1
        rfdata.echoMModeData = SortRF(echoMModeData, ML_Capture, ML_Actual, CRE, isVoyager)

    # Find color data
    color_index = np.zeros(xmitEvents).astype(bool)
    for i in range(len(DataType_COLOR)):
        index = rfdata.headerInfo.Data_Type == DataType_COLOR[i]
        color_index = np.bitwise_or(color_index, index)
    
    if (sum(color_index)>0):
        colorData = pruneData(rfdata.lineData[:,color_index], rfdata.lineHeader[:,color_index], ML_Capture)
        if (Tap_Point == 0 or Tap_Point == 1):
            ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrInCf*rfdata.dbParams.elevationMultilineFactorCf
        else:
            ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrOutCf*rfdata.dbParams.elevationMultilineFactorCf
        print("\t\tColor_ML:\t{0}x\n".format(ML_Actual))
        CRE = 1
        rfdata.colorData = SortRF(colorData, ML_Capture, ML_Actual, CRE, isVoyager)

        pkt = rfdata.dbParams.linesPerEnsCd
        nlv = rfdata.dbParams.ensPerSeqCf
        grp = rfdata.dbParams.numCfCols/rfdata.dbParams.ensPerSeqCf
        depth = rfdata.colorData.shape[0]

        # Extract and rearrange flow RF data
        frm = np.floor(rfdata.colorData.shape[2]/(nlv*pkt*grp)) # whole frames
        if frm == 0:
            warnings.warn("Cannot fully parse color data. RF capture does not contain at least one whole color frame.")
            frm = 1
            grp = np.floor(rfdata.colorData.shape[2]/(nlv*pkt))
        rfdata.colorData = rfdata.colorData[:,:,0:pkt*nlv*grp*frm-1]
        rfdata.colorData = np.reshape(rfdata.colorData, [depth, ML_Actual, nlv, pkt, grp, frm])
        rfdata.colorData = np.transpose(rfdata.colorData, (0,3,1,2,4,5))

    # Find Color MMode Data
    colorMMode_index = np.zeros(xmitEvents).astype(bool)
    for i in range(len(DataType_ColorMMode)):
        index = rfdata.headerInfo.Data_Type == DataType_ColorMMode[i]
        colorMMode_index = np.bitwise_or(colorMMode_index, index)
    
    if sum(colorMMode_index) > 0:
        colorMModeData = pruneData(rfdata.lineData[:,colorMMode_index], rfdata.lineHeader[:,colorMMode_index], ML_Capture)
        ML_Actual = 1
        CRE = 1
        rfdata.colorMModeData = SortRF(colorMModeData, ML_Capture, ML_Actual, CRE, isVoyager)
    
    # Find CW Doppler Data
    cw_index = np.zeros(xmitEvents).astype(bool)
    index = rfdata.headerInfo.Data_Type == DataType_CW
    cw_index = np.bitwise_or(cw_index, index)

    if (sum(cw_index) > 0):
        cwData = pruneData(rfdata.lineData[:,cw_index], rfdata.lineDeader[:,cw_index], ML_Capture)
        ML_Actual = 1
        CRE = 1
        rfdata.cwData = SortRF(cwData, ML_Capture, ML_Actual, CRE, isVoyager)

    # Find PW Doppler Data
    pw_index = np.zeros(xmitEvents).astype(bool)
    for i in range(len(DataType_PW)):
        index = rfdata.headerInfo.Data_Type == DataType_PW[i]
        pw_index = np.bitwise_or(pw_index, index)

    if (sum(cw_index) > 0):
        pwData = pruneData(rfdata.lineData[:,pw_index], rfdata.lineDeader[:,pw_index], ML_Capture)
        ML_Actual = 1
        CRE = 1
        rfdata.cwData = SortRF(pwData, ML_Capture, ML_Actual, CRE, isVoyager)

    # Find Dummy Data
    dummy_index = np.zeros(xmitEvents).astype(bool)
    for i in range(len(DataType_Dummy)):
        index = rfdata.headerInfo.Data_Type == DataType_Dummy[i]
        dummy_index = np.bitwise_or(dummy_index, index)

    if sum(dummy_index)>0:
        dummyData = pruneData(rfdata.lineData[:, dummy_index], rfdata.lineHeader[:, dummy_index], ML_Capture)
        ML_Actual = 2
        CRE = 1
        rfdata.dummyData = SortRF(dummyData, ML_Capture, ML_Actual, CRE, isVoyager)

    # Find Shearwave Data
    swi_index = np.zeros(xmitEvents).astype(bool)
    for i in range(len(DataType_SWI)):
        index = rfdata.headerInfo.Data_Type == DataType_SWI[i]
        swi_index = np.bitwise_or(swi_index, index)
    
    if sum(swi_index) > 0:
        swiData = pruneData(rfdata.lineData[:,swi_index], rfdata.lineHeader[:,swi_index], ML_Capture)
        ML_Actual = ML_Capture
        CRE = 1
        rfdata.swiData = SortRF(swiData, ML_Capture, ML_Actual, CRE, isVoyager)

    # Find Misc Data
    misc_index = np.zeros(xmitEvents).astype(bool)
    for i in range(len(DataType_Misc)):
        index = rfdata.headerInfo.Data_Type == DataType_Misc[i]
        misc_index = np.bitwise_or(misc_index, index)
    
    if sum(misc_index) > 0:
        miscData = pruneData(rfdata.lineData[:,misc_index], rfdata.lineHeader[:,misc_index], ML_Capture)
        ML_Actual = ML_Capture
        CRE = 1
        rfdata.miscData = SortRF(miscData, ML_Capture, ML_Actual, CRE, isVoyager)

    print (str("Elapsed time is " + str(-1*(start_time - datetime.now())) + " seconds."))

    # Clean up empty fields in struct
    print("Done")

    return rfdata


def philipsRfParser(filepath: str, ML_out=2, ML_in=32, used_os=2256) -> np.ndarray:
    """Parse Philips RF data file, save as .mat file, and return shape of data."""
    rf = parseRF(filepath, 0, 2000)

    if (rf.headerInfo.Line_Index[249] == rf.headerInfo.Line_Index[250]):
        rf.lineData = rf.lineData[:,np.arange(2, rf.lineData.shape[1], 2)]
    else:
        rf.lineData = rf.lineData[:,np.arange(1, rf.lineData.shape[1], 2)]

    txBeamperFrame = np.array(rf.dbParams.num2DCols).flat[0]
    NumSonoCTAngles = rf.dbParams.numOfSonoCTAngles2dActual[0]
    
    # # Calculated parameters 
    numFrame = int(np.floor(rf.lineData.shape[1]/txBeamperFrame/NumSonoCTAngles))
    multilinefactor = ML_in
    pt = int(np.floor((rf.lineData.shape[0]-used_os)/multilinefactor))

    rftemp_all_harm = np.zeros((pt,ML_out*txBeamperFrame))
    rftemp_all_fund = np.zeros((pt,ML_out*txBeamperFrame))
    rf_data_all_harm = np.zeros((numFrame,NumSonoCTAngles,pt,ML_out*txBeamperFrame))
    rf_data_all_fund = np.zeros((numFrame,NumSonoCTAngles,pt,ML_out*txBeamperFrame))

    for k0 in range(numFrame):
        for k1 in range(NumSonoCTAngles):
            for k2 in range(txBeamperFrame):
                bi = k0*txBeamperFrame*NumSonoCTAngles+k1*txBeamperFrame+k2
                temp = np.transpose(np.reshape(rf.lineData[used_os+np.arange(pt*multilinefactor),bi],(multilinefactor,pt), order='F'))
                rftemp_all_harm[:,np.arange(ML_out)+(k2*ML_out)] = temp[:,[0,2]]
                rftemp_all_fund[:,np.arange(ML_out)+(k2*ML_out)] = temp[:,[9,11]]

            rf_data_all_harm[k0][k1] = rftemp_all_harm
            rf_data_all_fund[k0][k1] = rftemp_all_fund

    # Save data
    destination = str(filepath[:-3] + '.mat')
    contents = {}
    contents['echoData'] = rf.echoData[0]
    contents['lineData'] = rf.lineData
    contents['lineHeader'] = rf.lineHeader
    contents['headerInfo'] = rf.headerInfo
    contents['dbParams'] = rf.dbParams
    contents['rf_data_all_fund'] = rf_data_all_fund
    contents['rf_data_all_harm'] = rf_data_all_harm
    contents['NumFrame'] = numFrame
    contents['NumSonoCTAngles'] = NumSonoCTAngles
    contents['pt'] = pt
    contents['multilinefactor'] = multilinefactor
    if len(rf.echoData[1]):
        contents['echoData1'] = rf.echoData[1]
    if len(rf.echoData[2]):
        contents['echoData2'] = rf.echoData[2]
    if len(rf.echoData[3]):
        contents['echoData3'] = rf.echoData[3]
    if hasattr(rf, 'echoMModeData'):
        contents['echoMModeData'] = rf.echoMModeData
    if hasattr(rf, 'miscData'):
        contents['miscData'] = rf.miscData
    
    if os.path.exists(destination):
        os.remove(destination)
    savemat(destination, contents)
    
    return np.array(rf_data_all_fund).shape

if __name__ == "__main__":
    philipsRfParser("/Users/davidspector/Downloads/parserRF_pywrap-2-2/rfCapture_20220511_144204.rf")