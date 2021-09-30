/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2016, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TComPattern.cpp
    \brief    neighbouring pixel access classes
*/

#include "TComPic.h"
#include "TComPattern.h"
#include "TComDataCU.h"
#include "TComTU.h"
#include "Debug.h"
#include "TComPrediction.h"

#include "TComTensorflow.h"
#include <cmath>
//nntra
#include <fstream>


#define NOMINMAX
#define COMPILER_MSVC

bool nnBestModeFlag = false;
bool nnBestModeChromaFlag = false;
int nnMode = 0;

int model = 1;
bool pixel_norm = false;

Pel multiRec[(MAX_CU_SIZE * 2 + MAX_CU_SIZE) * MAX_CU_SIZE + (MAX_CU_SIZE * 2) * MAX_CU_SIZE] = { 128 };

extern bool finalBestModeFlag;
extern bool finalBestChromaModeFlag;

int nnBestMode[32400] = { 0 };

void nnPred(Pel *recPixel, int nnRefPel, int outPelNum,  Pel *predPixel, Pel *hmPredPixel, int ch)
{

	NNPredict predict;
	//cout << predict.get_a() << endl;

	if ((ch == 0) && (outPelNum==4096) )
		cout << outPelNum << endl;

	PRECISION *featureMapIn = (PRECISION*)malloc(nnRefPel * sizeof(PRECISION));
	PRECISION *featureMapMid4 = (PRECISION*)malloc(outPelNum*sizeof(PRECISION));
	PRECISION *featureMapOut = (PRECISION*)malloc(outPelNum*sizeof(PRECISION));

	tensorflow::Session* local_session;

	int tuSize = 0;
	if (outPelNum == 16) //tu4
	{
		tuSize = 0;
		local_session = predict.session_4[nnMode - 1];
	}
	else if (outPelNum == 64) //tu8
	{
		tuSize = 1;
		local_session = predict.session_8[nnMode - 1];
	}
	else if (outPelNum == 256) //tu16
	{
		tuSize = 2;
		local_session = predict.session_16[nnMode - 1];
	}
	else if (outPelNum == 1024)
	{

		tuSize = 3;
		local_session = predict.session_32[nnMode - 1];
	}
	else
	{
		cout << "size not correct" << endl;
		assert(0);
	}

	for (int i = 0; i < nnRefPel; i++)
	{
		//featureMapIn[i] = (double(recPixel[i]) / (pixel_norm ? double(255):double(1) ) ) - ave;
		featureMapIn[i] = (PRECISION(recPixel[i]) / (pixel_norm ? PRECISION(255) : PRECISION(1)));
	}
	
	predict.run_session(featureMapIn, featureMapMid4, nnRefPel, local_session);

	if (model == 1)
	{
		for (int i = 0; i < outPelNum; i++)
		{
			featureMapOut[i] = featureMapMid4[i];
		}
	}
	else
	{
		cout << "define the model" << endl;
		assert(0);
  	}

	for (int i = 0; i < outPelNum; i++)
	{
		predPixel[i] = Pel(featureMapOut[i]);
	}


	free(featureMapIn);
	free(featureMapMid4);
	free(featureMapOut);
}

//! \ingroup TLibCommon
//! \{

// Forward declarations

/// padding of unavailable reference samples for intra prediction
Void fillReferenceSamples( const Int bitDepth, 
#if O0043_BEST_EFFORT_DECODING
                           const Int bitDepthDelta, 
#endif
                           const Pel* piRoiOrigin, 
                                 Pel* piIntraTemp,
                           const Bool* bNeighborFlags,
                           const Int iNumIntraNeighbor, 
                           const Int unitWidth, 
                           const Int unitHeight, 
                           const Int iAboveUnits, 
                           const Int iLeftUnits,
                           const UInt uiWidth, 
                           const UInt uiHeight, 
                           const Int iPicStride
						         );

#if ApplyIntraFCN
Void fillExtendedReferenceSamples(
	const Int offset,
	const Int bitDepth,
	const Pel* piRoiOrigin,
	Pel* piIntraTemp,
	const Bool* bNeighborFlags,
	const Int iNumIntraNeighbor,
	const Int unitWidth,
	const Int unitHeight,
	const Int iAboveUnits,
	const Int iLeftUnits,
	const UInt uiWidth,
	const UInt uiHeight,
	const Int iPicStride,
	const Bool IsAboveRightSpecialAvailable,
	const Bool IsBelowLeftSpecialAvailable
);

//uiZorderIdxInPart is relate to a CTU
Bool CheckSpecialAvailable(TComDataCU* pcCU, UInt uiZorderIdxInPart, UInt unitOffset, UInt unitIndex, Bool IsAbove)
{
	Int CurY = pcCU->getPic()->getCtu(pcCU->getCtuRsAddr())->getCUPelY() + g_auiRasterToPelY[g_auiZscanToRaster[uiZorderIdxInPart]];
	Int CurX = pcCU->getPic()->getCtu(pcCU->getCtuRsAddr())->getCUPelX() + g_auiRasterToPelX[g_auiZscanToRaster[uiZorderIdxInPart]];

	Int BaseUnitSize = pcCU->getPic()->getMinCUWidth();
	Int CTUSize = pcCU->getSlice()->getSPS()->getMaxCUWidth();
	const UInt numPartInCtuWidth = pcCU->getPic()->getNumPartInCtuWidth();
	//check_jiahao
	//assert(pcCU->getPic()->getMinCUWidth() == (pcCU->getSlice()->getSPS()->getMaxCUWidth() >> pcCU->getSlice()->getSPS()->getMaxTotalCUDepth()));
	//assert(pcCU->getPic()->getMinCUWidth() == pcCU->getPic()->getMinCUHeight());
	//assert(BaseUnitSize == 4);
	//assert(numPartInCtuWidth == 16);

	if (IsAbove)//above or above-right
	{
		Int TargetY = CurY - unitOffset * BaseUnitSize - BaseUnitSize;
		Int TargetX = CurX - unitOffset * BaseUnitSize + unitIndex * BaseUnitSize;
		if (TargetX >= pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples() || TargetY < 0 || TargetX < 0)
		{
			return false;
		}
		// may occur the above-left
		if (TargetX < CurX)
		{
			return true;
		}

		// is same CTU row?
		if ((CurY >> 6) != (TargetY >> 6))
		{
			return true;
		}
		else // same CTU row
		{
			// is same CTU column?
			if ((CurX >> 6) != (TargetX >> 6))
			{
				return false;
			}
			else //same CTU column
			{
				Int TargetAbsZorder = ((TargetY&(CTUSize - 1)) >> 2)*numPartInCtuWidth + ((TargetX&(CTUSize - 1)) >> 2);
				TargetAbsZorder = g_auiRasterToZscan[TargetAbsZorder];

				//check_jiahao
				//assert(TargetAbsZorder == g_auiRasterToZscan[g_auiZscanToRaster[uiZorderIdxInPart] - (unitOffset + 1) * numPartInCtuWidth - unitOffset + unitIndex]);
				if (TargetAbsZorder < uiZorderIdxInPart)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
		}
	}
	else // left
	{
		Int TargetY = CurY - unitOffset * BaseUnitSize + unitIndex * BaseUnitSize;
		Int TargetX = CurX - unitOffset * BaseUnitSize - BaseUnitSize;
		if (TargetY >= pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples() || TargetY < 0 || TargetX < 0)
		{
			return false;
		}
		// may occur the above-left
		if (TargetY < CurY)
		{
			return true;
		}

		// is same CTU row?
		if ((CurY >> 6) != (TargetY >> 6))
		{
			return false;
		}
		else // same CTU row
		{
			// is same CTU column?
			if ((CurX >> 6) != (TargetX >> 6))
			{
				return true;
			}
			else //same CTU column
			{
				Int TargetAbsZorder = ((TargetY&(CTUSize - 1)) >> 2)*numPartInCtuWidth + ((TargetX&(CTUSize - 1)) >> 2);
				TargetAbsZorder = g_auiRasterToZscan[TargetAbsZorder];


				//check_jiahao
				//assert(TargetAbsZorder == g_auiRasterToZscan[g_auiZscanToRaster[uiZorderIdxInPart] + numPartInCtuWidth * (unitIndex - unitOffset) - 1 - unitOffset]);
				if (TargetAbsZorder < uiZorderIdxInPart)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
		}
	}

	assert(0);
	return false;
}
#endif

/// constrained intra prediction
Bool  isAboveLeftAvailable  ( const TComDataCU* pcCU, UInt uiPartIdxLT );
Int   isAboveAvailable      ( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxRT, Bool* bValidFlags );
Int   isLeftAvailable       ( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxLB, Bool* bValidFlags );
Int   isAboveRightAvailable ( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxRT, Bool* bValidFlags );
Int   isBelowLeftAvailable  ( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxLB, Bool* bValidFlags );


// ====================================================================================================================
// Public member functions (TComPatternParam)
// ====================================================================================================================

/** 
 \param  piTexture     pixel data
 \param  iRoiWidth     pattern width
 \param  iRoiHeight    pattern height
 \param  iStride       buffer stride
 \param  bitDepth      bit depth
 */
Void TComPatternParam::setPatternParamPel ( Pel* piTexture,
                                           Int iRoiWidth,
                                           Int iRoiHeight,
                                           Int iStride,
                                           Int bitDepth
                                           )
{
  m_piROIOrigin    = piTexture;
  m_iROIWidth      = iRoiWidth;
  m_iROIHeight     = iRoiHeight;
  m_iPatternStride = iStride;
  m_bitDepth       = bitDepth;
}

// ====================================================================================================================
// Public member functions (TComPattern)
// ====================================================================================================================

Void TComPattern::initPattern (Pel* piY,
                               Int iRoiWidth,
                               Int iRoiHeight,
                               Int iStride,
                               Int bitDepthLuma)
{
  m_cPatternY. setPatternParamPel( piY,  iRoiWidth, iRoiHeight, iStride, bitDepthLuma);
}

#if ApplyIntraFCN
void TComPrediction::getFurtherRefLine(bool curIsChroma, int lineIndex, Pel* piRoiOrigin, int iPicStride, TComDataCU *pcCU, UInt uiPartIdxLT, Int iTUWidthInUnits, Int iTUHeightInUnits, int iUnitWidth, int iUnitHeight)
{
	int comBasicSize = 4;
	if (curIsChroma)
	{
		comBasicSize = 2;
	}

	const Int unitOffset = lineIndex / comBasicSize;
	const Int  iAboveUnits = (iTUWidthInUnits + unitOffset) << 1;
	const Int  iLeftUnits = (iTUHeightInUnits + unitOffset) << 1;
	Pel tempCurLine[79 * 79];
	Bool  bNeighborFlags[5 * MAX_NUM_PART_IDXS_IN_CTU_WIDTH];
	Int   iNumIntraNeighbor = 0;
	//above left
	{
		Int CurY = pcCU->getPic()->getCtu(pcCU->getCtuRsAddr())->getCUPelY() + g_auiRasterToPelY[g_auiZscanToRaster[uiPartIdxLT]];
		Int CurX = pcCU->getPic()->getCtu(pcCU->getCtuRsAddr())->getCUPelX() + g_auiRasterToPelX[g_auiZscanToRaster[uiPartIdxLT]];

		Int topLeftY, topLeftX;
		if (curIsChroma)
		{
			topLeftY = CurY - (lineIndex + 1) * 2;
			topLeftX = CurX - (lineIndex + 1) * 2;
		}
		else
		{
			topLeftY = CurY - (lineIndex + 1);
			topLeftX = CurX - (lineIndex + 1);
		}

		if (topLeftY >= 0 && topLeftX >= 0)
		{
			bNeighborFlags[iLeftUnits] = true;
		}
		else
		{
			bNeighborFlags[iLeftUnits] = false;
		}


		iNumIntraNeighbor += bNeighborFlags[iLeftUnits] ? 1 : 0;
	}
	// above and above right
	{
		for (Int k = 0; k < iAboveUnits; k++)
		{
			bNeighborFlags[iLeftUnits + 1 + k] = CheckSpecialAvailable(pcCU, uiPartIdxLT, unitOffset, k, true);
			iNumIntraNeighbor += bNeighborFlags[iLeftUnits + 1 + k] ? 1 : 0;
		}
	}
	// left and below left
	{
		for (Int k = 0; k < iLeftUnits; k++)
		{
			bNeighborFlags[iLeftUnits - 1 - k] = CheckSpecialAvailable(pcCU, uiPartIdxLT, unitOffset, k, false);
			iNumIntraNeighbor += bNeighborFlags[iLeftUnits - 1 - k] ? 1 : 0;
		}
	}
	Int offset = lineIndex % comBasicSize;

	const UInt         uiROIWidthTemp = iAboveUnits * iUnitWidth + 2 * offset + 1;
	const UInt         uiROIHeightTemp = iLeftUnits * iUnitHeight + 2 * offset + 1;

	//assert(uiROIWidthTemp == (2 * (iTUWidthInUnits*iUnitWidth + lineIndex) + 1));
	//assert(uiROIHeightTemp == (2 * (iTUHeightInUnits*iUnitHeight + lineIndex) + 1));

	Pel *piRoiOriginTemp = piRoiOrigin - unitOffset * iUnitHeight*iPicStride - unitOffset * iUnitWidth;

	fillExtendedReferenceSamples(offset, 8, piRoiOriginTemp, tempCurLine, bNeighborFlags, iNumIntraNeighbor, iUnitWidth, iUnitHeight, iAboveUnits, iLeftUnits, uiROIWidthTemp, uiROIHeightTemp, iPicStride, false, false);


	Pel* taget = m_multiRefLines + (7 - lineIndex) * m_multiRefStride + (7 - lineIndex);
	for (int k = 0; k < uiROIWidthTemp - lineIndex; k++)
	{
		taget[k] = tempCurLine[k];
		taget[k*m_multiRefStride] = tempCurLine[k*uiROIWidthTemp];

		//assert(taget[k] >= 0 && taget[k] < 256);
		//assert(taget[k*m_multiRefStride] >= 0 && taget[k*m_multiRefStride] < 256);
	}


}
#endif

// TODO: move this function to TComPrediction.cpp.
Void TComPrediction::initIntraPatternChType( TComTU &rTu, const ComponentID compID, const Bool bFilterRefSamples DEBUG_STRING_FN_DECLARE(sDebug))
{
  const ChannelType chType    = toChannelType(compID);

  TComDataCU *pcCU=rTu.getCU();
  const TComSPS &sps = *(pcCU->getSlice()->getSPS());
  const UInt uiZorderIdxInPart=rTu.GetAbsPartIdxTU(); //tu in cu
  const UInt uiTuWidth        = rTu.getRect(compID).width;
  const UInt uiTuHeight       = rTu.getRect(compID).height;
  const UInt uiTuWidth2       = uiTuWidth  << 1;
  const UInt uiTuHeight2      = uiTuHeight << 1;

  const Int  iBaseUnitSize    = sps.getMaxCUWidth() >> sps.getMaxTotalCUDepth(); //4x4 
  const Int  iUnitWidth       = iBaseUnitSize  >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(compID);
  const Int  iUnitHeight      = iBaseUnitSize  >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(compID);
  const Int  iTUWidthInUnits  = uiTuWidth  / iUnitWidth; // how many basic units (4x4 in the luma case) in the tu
  const Int  iTUHeightInUnits = uiTuHeight / iUnitHeight;
  const Int  iAboveUnits      = iTUWidthInUnits  << 1;
  const Int  iLeftUnits       = iTUHeightInUnits << 1;
  const Int  bitDepthForChannel = sps.getBitDepth(chType);

  assert(iTUHeightInUnits > 0 && iTUWidthInUnits > 0);

  const Int  iPartIdxStride   = pcCU->getPic()->getNumPartInCtuWidth(); //ctu size/4
  const UInt uiPartIdxLT      = pcCU->getZorderIdxInCtu() + uiZorderIdxInPart; //the left top 4x4 of tu in ctu
  const UInt uiPartIdxRT      = g_auiRasterToZscan[ g_auiZscanToRaster[ uiPartIdxLT ] +   iTUWidthInUnits  - 1                   ];
  const UInt uiPartIdxLB      = g_auiRasterToZscan[ g_auiZscanToRaster[ uiPartIdxLT ] + ((iTUHeightInUnits - 1) * iPartIdxStride)];

  //pcCU->getZorderIdxInCtu(): cu in ctu
  //uiZorderIdxInPart : tu in cu

  Int   iPicStride = pcCU->getPic()->getStride(compID); //pic width + margin

  Bool  bNeighborFlags[4 * MAX_NUM_PART_IDXS_IN_CTU_WIDTH + 1];
  Int   iNumIntraNeighbor = 0;

  //cout << "cu in ctu: " << pcCU->getZorderIdxInCtu() << " tu in cu: " << uiZorderIdxInPart << endl;

  bNeighborFlags[iLeftUnits] = isAboveLeftAvailable( pcCU, uiPartIdxLT );
  iNumIntraNeighbor += bNeighborFlags[iLeftUnits] ? 1 : 0;
  iNumIntraNeighbor  += isAboveAvailable     ( pcCU, uiPartIdxLT, uiPartIdxRT, (bNeighborFlags + iLeftUnits + 1)                    );
  iNumIntraNeighbor  += isAboveRightAvailable( pcCU, uiPartIdxLT, uiPartIdxRT, (bNeighborFlags + iLeftUnits + 1 + iTUWidthInUnits ) );
  iNumIntraNeighbor  += isLeftAvailable      ( pcCU, uiPartIdxLT, uiPartIdxLB, (bNeighborFlags + iLeftUnits - 1)                    );
  iNumIntraNeighbor  += isBelowLeftAvailable ( pcCU, uiPartIdxLT, uiPartIdxLB, (bNeighborFlags + iLeftUnits - 1 - iTUHeightInUnits) );

  const UInt         uiROIWidth  = uiTuWidth2+1;
  const UInt         uiROIHeight = uiTuHeight2+1;

  assert(uiROIWidth*uiROIHeight <= m_iYuvExtSize);

#if DEBUG_STRING
  std::stringstream ss(stringstream::out);
#endif

  {
    Pel *piIntraTemp   = m_piYuvExt[compID][PRED_BUF_UNFILTERED]; //the second dimension: 0 for unfiltered, 1 for filtered, piIntraTemp[0] is left-top 1 pixel
    Pel *piRoiOrigin = pcCU->getPic()->getPicYuvRec()->getAddr(compID, pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu()+uiZorderIdxInPart);

#if O0043_BEST_EFFORT_DECODING
    const Int  bitDepthForChannelInStream = sps.getStreamBitDepth(chType);
    fillReferenceSamples (bitDepthForChannelInStream, bitDepthForChannelInStream - bitDepthForChannel,
#else
    fillReferenceSamples (bitDepthForChannel,
#endif
                          piRoiOrigin, piIntraTemp, bNeighborFlags, iNumIntraNeighbor,  iUnitWidth, iUnitHeight, iAboveUnits, iLeftUnits,
                          uiROIWidth, uiROIHeight, iPicStride);

#if ApplyIntraFCN

	if ( /* pcCU->getIsNetworkFlag(uiZorderIdxInPart) && */ (pcCU->getCUPelX() >= 0 ) && ( pcCU->getCUPelY() >= 0 ) && (uiTuWidth <= 32) )
	{
		m_multiRefStride = -1;
		//Int Topx = pcCU->getCUPelX() + g_auiRasterToPelX[g_auiZscanToRaster[uiZorderIdxInPart]];
		//Int Topy = pcCU->getCUPelY() + g_auiRasterToPelY[g_auiZscanToRaster[uiZorderIdxInPart]];
		/*if (isLuma(compID))
		{
		  assert(Topx >= 8 && Topy >= 8);
		}*/

		{
			//assert(bitDepthForChannel == 8);

			memset(m_multiRefLines, 0, sizeof(Pel) * 72 * 72);
			m_multiRefStride = uiTuWidth2 + 8;


			///nearest reference line
			Pel* taget = m_multiRefLines + 7 * m_multiRefStride + 7;
			for (int k = 0; k < uiROIWidth; k++)
			{
				taget[k] = piIntraTemp[k];
				taget[k*m_multiRefStride] = piIntraTemp[k*uiROIWidth];

				//assert(taget[k] >= 0 && taget[k] < 256);
				//assert(taget[k*m_multiRefStride] >= 0 && taget[k*m_multiRefStride] < 256);
			}


			///1st-7rd reference line
			for (int lineIndex = 1; lineIndex < 8; lineIndex++)
			{
				getFurtherRefLine(isChroma(compID), lineIndex, piRoiOrigin, iPicStride, pcCU, uiPartIdxLT, iTUWidthInUnits, iTUHeightInUnits, iUnitWidth, iUnitHeight);
			}
  }

}
#endif

#if DEBUG_STRING
    if (DebugOptionList::DebugString_Pred.getInt()&DebugStringGetPredModeMask(MODE_INTRA))
    {
      ss << "###: generating Ref Samples for channel " << compID << " and " << rTu.getRect(compID).width << " x " << rTu.getRect(compID).height << "\n";
      for (UInt y=0; y<uiROIHeight; y++)
      {
        ss << "###: - ";
        for (UInt x=0; x<uiROIWidth; x++)
        {
          if (x==0 || y==0)
          {
            ss << piIntraTemp[y*uiROIWidth + x] << ", ";
//          if (x%16==15) ss << "\nPart size: ~ ";
          }
        }
        ss << "\n";
      }
    }
#endif

    if (bFilterRefSamples)
	{
		//cout << " do " << endl;
      // generate filtered intra prediction samples

            Int          stride    = uiROIWidth;
      const Pel         *piSrcPtr  = piIntraTemp                           + (stride * uiTuHeight2); // bottom left
            Pel         *piDestPtr = m_piYuvExt[compID][PRED_BUF_FILTERED] + (stride * uiTuHeight2); // bottom left

      //------------------------------------------------

      Bool useStrongIntraSmoothing = isLuma(chType) && sps.getUseStrongIntraSmoothing();

      const Pel bottomLeft = piIntraTemp[stride * uiTuHeight2];
      const Pel topLeft    = piIntraTemp[0];
      const Pel topRight   = piIntraTemp[uiTuWidth2];

      if (useStrongIntraSmoothing)
      {
#if O0043_BEST_EFFORT_DECODING
        const Int  threshold     = 1 << (bitDepthForChannelInStream - 5);
#else
        const Int  threshold     = 1 << (bitDepthForChannel - 5);
#endif
        const Bool bilinearLeft  = abs((bottomLeft + topLeft ) - (2 * piIntraTemp[stride * uiTuHeight])) < threshold; //difference between the
        const Bool bilinearAbove = abs((topLeft    + topRight) - (2 * piIntraTemp[         uiTuWidth ])) < threshold; //ends and the middle
        if ((uiTuWidth < 32) || (!bilinearLeft) || (!bilinearAbove))
        {
          useStrongIntraSmoothing = false;
        }
      }

      *piDestPtr = *piSrcPtr; // bottom left is not filtered
      piDestPtr -= stride;
      piSrcPtr  -= stride;

      //------------------------------------------------

      //left column (bottom to top)

      if (useStrongIntraSmoothing)
      {
        const Int shift = g_aucConvertToBit[uiTuHeight] + 3; //log2(uiTuHeight2)

        for(UInt i=1; i<uiTuHeight2; i++, piDestPtr-=stride)
        {
          *piDestPtr = (((uiTuHeight2 - i) * bottomLeft) + (i * topLeft) + uiTuHeight) >> shift;
        }

        piSrcPtr -= stride * (uiTuHeight2 - 1);
      }
      else
      {
        for(UInt i=1; i<uiTuHeight2; i++, piDestPtr-=stride, piSrcPtr-=stride)
        {
          *piDestPtr = ( piSrcPtr[stride] + 2*piSrcPtr[0] + piSrcPtr[-stride] + 2 ) >> 2;
        }
      }

      //------------------------------------------------

      //top-left

      if (useStrongIntraSmoothing)
      {
        *piDestPtr = piSrcPtr[0];
      }
      else
      {
        *piDestPtr = ( piSrcPtr[stride] + 2*piSrcPtr[0] + piSrcPtr[1] + 2 ) >> 2;
      }
      piDestPtr += 1;
      piSrcPtr  += 1;

      //------------------------------------------------

      //top row (left-to-right)

      if (useStrongIntraSmoothing)
      {
        const Int shift = g_aucConvertToBit[uiTuWidth] + 3; //log2(uiTuWidth2)

        for(UInt i=1; i<uiTuWidth2; i++, piDestPtr++)
        {
          *piDestPtr = (((uiTuWidth2 - i) * topLeft) + (i * topRight) + uiTuWidth) >> shift;
        }

        piSrcPtr += uiTuWidth2 - 1;
      }
      else
      {
        for(UInt i=1; i<uiTuWidth2; i++, piDestPtr++, piSrcPtr++)
        {
          *piDestPtr = ( piSrcPtr[1] + 2*piSrcPtr[0] + piSrcPtr[-1] + 2 ) >> 2;
        }
      }

      //------------------------------------------------

      *piDestPtr=*piSrcPtr; // far right is not filtered

#if DEBUG_STRING
    if (DebugOptionList::DebugString_Pred.getInt()&DebugStringGetPredModeMask(MODE_INTRA))
    {
      ss << "###: filtered result for channel " << compID <<"\n";
      for (UInt y=0; y<uiROIHeight; y++)
      {
        ss << "###: - ";
        for (UInt x=0; x<uiROIWidth; x++)
        {
          if (x==0 || y==0)
          {
            ss << m_piYuvExt[compID][PRED_BUF_FILTERED][y*uiROIWidth + x] << ", ";
//          if (x%16==15) ss << "\nPart size: ~ ";
          }
        }
        ss << "\n";
      }
    }
#endif


    }
  }
  DEBUG_STRING_APPEND(sDebug, ss.str())
}


#if ApplyIntraFCN
Void fillExtendedReferenceSamples(
	const Int offset,
	const Int bitDepth,
	const Pel* piRoiOrigin,
	Pel* piIntraTemp,
	const Bool* bNeighborFlags,
	const Int iNumIntraNeighbor,
	const Int unitWidth,
	const Int unitHeight,
	const Int iAboveUnits,
	const Int iLeftUnits,
	const UInt uiWidth,
	const UInt uiHeight,
	const Int iPicStride,
	const Bool IsAboveRightSpecialAvailable,
	const Bool IsBelowLeftSpecialAvailable)
{
	//assert(uiWidth == uiHeight);
	//assert(uiWidth == iAboveUnits*unitWidth + 2 * offset + 1);

	//assert(unitWidth == 4);

	const Pel* piRoiTemp;
	Int  i, j;
	Int  iDCValue = 1 << (bitDepth - 1);
	const Int iTotalUnits = iAboveUnits + iLeftUnits + 1; //+1 for top-left

	if (iNumIntraNeighbor == 0)
	{
		// Fill border with DC value
		for (i = 0; i < uiWidth; i++)
		{
			piIntraTemp[i] = iDCValue;
		}
		for (i = 1; i < uiHeight; i++)
		{
			piIntraTemp[i*uiWidth] = iDCValue;
		}
		return;
	}
	else if (iNumIntraNeighbor == iTotalUnits)
	{
		// Fill top-left
		piIntraTemp[0] = piRoiOrigin[-(offset + 1) * iPicStride - offset - 1];

		// Fill  top and top right with rec. samples
		piRoiTemp = piRoiOrigin - (offset + 1) * iPicStride - (offset + 1);
		for (i = 1; i < uiWidth - offset; i++)
		{
			piIntraTemp[i] = piRoiTemp[i];
		}

		// Fill left and below left border with rec. samples
		piRoiTemp = piRoiOrigin - (offset + 1) * iPicStride - (offset + 1);
		for (i = 1; i < uiHeight - offset; i++)
		{
			piIntraTemp[i*uiWidth] = piRoiTemp[i*iPicStride];
		}
	}
	else // reference samples are partially available
	{
		// all above units have "unitWidth" samples each, all left/below-left units have "unitHeight" samples each
		const Int NumTopLeft = 2 * offset + 1;
		const Int  iTotalSamples = (iLeftUnits * unitHeight) + (iAboveUnits  * unitWidth) + NumTopLeft;

		Pel  piIntraLine[5 * MAX_CU_SIZE];
		//assert(iTotalSamples <= 5 * MAX_CU_SIZE);

		Pel  *piIntraLineTemp;
		const Bool *pbNeighborFlags;


		// Initialize
		for (i = 0; i < iTotalSamples; i++)
		{
			piIntraLine[i] = iDCValue;
		}


		// Fill top-left sample
		piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight);
		pbNeighborFlags = bNeighborFlags + iLeftUnits;
		if (*pbNeighborFlags)
		{
			for (i = 0; i < NumTopLeft; i++)
			{
				if (i <= offset)
				{
					piIntraLineTemp[i] = piRoiOrigin[-(i + 1)*iPicStride - offset - 1];
				}
				else
				{
					piIntraLineTemp[i] = piRoiOrigin[-(offset + 1)*iPicStride - offset - 1 + (i - offset)];
				}
			}
			// assert(i == (2 * offset + 1));
		}

		// Fill left & below-left samples (downwards)
		piRoiTemp = piRoiOrigin - offset - 1;
		piIntraLineTemp--;
		pbNeighborFlags--;

		for (j = 0; j < iLeftUnits; j++)
		{
			if (*pbNeighborFlags)
			{
				for (i = 0; i < unitHeight; i++)
				{
					piIntraLineTemp[-i] = piRoiTemp[i*iPicStride];
				}
			}
			piRoiTemp += unitHeight * iPicStride;
			piIntraLineTemp -= unitHeight;
			pbNeighborFlags--;
		}

		// Fill above & above-right samples (left-to-right) (each unit has "unitWidth" samples)
		piRoiTemp = piRoiOrigin - (offset + 1)*iPicStride;
		// offset line buffer by iNumUints2*unitHeight (for left/below-left) + unitWidth (for above-left)
		piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight) + NumTopLeft;
		pbNeighborFlags = bNeighborFlags + iLeftUnits + 1;
		for (j = 0; j < iAboveUnits; j++)
		{
			if (*pbNeighborFlags)
			{
				for (i = 0; i < unitWidth; i++)
				{
					piIntraLineTemp[i] = piRoiTemp[i];
				}
			}
			piRoiTemp += unitWidth;
			piIntraLineTemp += unitWidth;
			pbNeighborFlags++;
		}

		// Pad reference samples when necessary
		Int iCurrJnit = 0;
		Pel  *piIntraLineCur = piIntraLine;


		if (!bNeighborFlags[0])
		{
			// very bottom unit of bottom-left; at least one unit will be valid.
			{
				Int   iNext = 1;
				while (iNext < iTotalUnits && !bNeighborFlags[iNext])
				{
					iNext++;
				}
				// assert(iNext < iTotalUnits);

				Pel *piIntraLineNext = piIntraLine + ((iNext <= iLeftUnits) ? (iNext * unitHeight) : (iLeftUnits*unitHeight + NumTopLeft + ((iNext - iLeftUnits - 1) * unitWidth)));
				const Pel refSample = *piIntraLineNext;
				// Pad unavailable samples with new value
				Int iNextOrTop = std::min<Int>(iNext, iLeftUnits);
				// fill left column
				while (iCurrJnit < iNextOrTop)
				{
					for (i = 0; i < unitHeight; i++)
					{
						piIntraLineCur[i] = refSample;
					}
					piIntraLineCur += unitHeight;
					iCurrJnit++;
				}
				//fill top left
				if (iCurrJnit < iNext)
				{
					//assert(iCurrJnit == iLeftUnits);

					for (i = 0; i < NumTopLeft; i++)
					{
						piIntraLineCur[i] = refSample;
					}
					piIntraLineCur += NumTopLeft;
					iCurrJnit++;
				}

				// fill top row
				while (iCurrJnit < iNext)
				{
					for (i = 0; i < unitWidth; i++)
					{
						piIntraLineCur[i] = refSample;
					}
					piIntraLineCur += unitWidth;
					iCurrJnit++;
				}
			}
		}
		//assert(bNeighborFlags[iCurrJnit]);

		// pad all other reference samples.
		while (iCurrJnit < iTotalUnits)
		{
			if (!bNeighborFlags[iCurrJnit]) // samples not available
			{
				Int numSamplesInCurrUnit;
				if (iCurrJnit < iLeftUnits)
				{
					numSamplesInCurrUnit = unitHeight;
				}
				else if (iCurrJnit == iLeftUnits)
				{
					numSamplesInCurrUnit = NumTopLeft;
				}
				else
				{
					numSamplesInCurrUnit = unitWidth;
				}

				const Pel refSample = *(piIntraLineCur - 1);
				for (i = 0; i < numSamplesInCurrUnit; i++)
				{
					piIntraLineCur[i] = refSample;
				}
				piIntraLineCur += numSamplesInCurrUnit;
				iCurrJnit++;

			}
			else
			{
				if (iCurrJnit < iLeftUnits)
				{
					piIntraLineCur += unitHeight;
				}
				else if (iCurrJnit == iLeftUnits)
				{
					piIntraLineCur += NumTopLeft;
				}
				else
				{
					piIntraLineCur += unitWidth;
				}
				iCurrJnit++;
			}
		}

		// Copy processed samples
		// top left
		piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight);
		for (i = 0; i < NumTopLeft; i++)
		{
			if (i <= offset)
			{
				piIntraTemp[uiWidth*(offset - i)] = piIntraLineTemp[i];
			}
			else
			{
				piIntraTemp[(i - offset)] = piIntraLineTemp[i];
			}
		}
		// top and top right samples
		piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight) + NumTopLeft;

		for (i = offset + 1; i < uiWidth - offset; i++)
		{
			piIntraTemp[i] = piIntraLineTemp[i - (offset + 1)];
		}

		// left and below left samples
		piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight) - 1;
		for (i = offset + 1; i < uiHeight - offset; i++)
		{
			piIntraTemp[i*uiWidth] = piIntraLineTemp[-(i - (offset + 1))];
		}
	}
	// process the two special samples 
	// above right
	if (IsAboveRightSpecialAvailable)
	{
		piRoiTemp = piRoiOrigin - (offset + 1) * iPicStride + iAboveUnits * unitWidth;

		for (i = 0; i < offset; i++)
		{
			piIntraTemp[uiWidth - offset + i] = piRoiTemp[i];
		}
	}
	else
	{
		for (i = 0; i < offset; i++)
		{
			piIntraTemp[uiWidth - offset + i] = piIntraTemp[uiWidth - offset - 1];
		}

	}
	//below left
	if (IsBelowLeftSpecialAvailable)
	{
		piRoiTemp = piRoiOrigin + (iLeftUnits*unitHeight) * iPicStride - (offset + 1);
		for (i = 0; i < offset; i++)
		{
			piIntraTemp[(uiHeight - offset + i)*uiWidth] = piRoiTemp[i*iPicStride];
		}
	}
	else
	{
		for (i = 0; i < offset; i++)
		{
			piIntraTemp[(uiHeight - offset + i)*uiWidth] = piIntraTemp[(uiHeight - offset - 1)*uiWidth];
		}

	}

}
#endif

Void fillReferenceSamples( const Int bitDepth, 
#if O0043_BEST_EFFORT_DECODING
                           const Int bitDepthDelta, 
#endif
                           const Pel* piRoiOrigin, 
                                 Pel* piIntraTemp,
                           const Bool* bNeighborFlags,
                           const Int iNumIntraNeighbor, 
                           const Int unitWidth, 
                           const Int unitHeight, 
                           const Int iAboveUnits, 
                           const Int iLeftUnits,
                           const UInt uiWidth, 
                           const UInt uiHeight, 
                           const Int iPicStride
						         )
{
  const Pel* piRoiTemp;
  Int  i, j;
  Int  iDCValue = 1 << (bitDepth - 1);
  const Int iTotalUnits = iAboveUnits + iLeftUnits + 1; //+1 for top-left

  // nntra
  int multi_lane = (uiWidth - 1) / 2;
  const Pel* nnRoiTemp;
  //int nnTusize = (uiHeight - 1) / 2;
  int nnTotalPixels = (uiWidth - 1 + multi_lane) * multi_lane + (uiHeight - 1) * multi_lane;

  //cout << nnTotalPixels << endl;

  Int nni, nnj;

  if (iNumIntraNeighbor == 0) // fill all the referenced pixels by dc
  {
    // Fill border with DC value
    for (i=0; i<uiWidth; i++)
    {
      piIntraTemp[i] = iDCValue;
    }
    for (i=1; i<uiHeight; i++)
    {
      piIntraTemp[i*uiWidth] = iDCValue;
    }

	// nntra
	for (nni = 0; nni < nnTotalPixels; nni++)
	{
		multiRec[nni] = iDCValue;
	}

  }
  else if (iNumIntraNeighbor == iTotalUnits)
  {
    // Fill top-left border and top and top right with rec. samples
    piRoiTemp = piRoiOrigin - iPicStride - 1; // point to the left-top 1 pixel in the rec yuv

    for (i=0; i<uiWidth; i++) // fill left-top, top, top-right with rec. samples
    {
#if O0043_BEST_EFFORT_DECODING
      piIntraTemp[i] = piRoiTemp[i] << bitDepthDelta;
#else
      piIntraTemp[i] = piRoiTemp[i];
#endif
    }

    // Fill left and below left border with rec. samples
    piRoiTemp = piRoiOrigin - 1;

    for (i=1; i<uiHeight; i++)
    {
#if O0043_BEST_EFFORT_DECODING
      piIntraTemp[i*uiWidth] = (*(piRoiTemp)) << bitDepthDelta;
#else
      piIntraTemp[i*uiWidth] = *(piRoiTemp);
#endif
      piRoiTemp += iPicStride;
    }

	// Fill for nntra
	// fill the left
	nnRoiTemp = piRoiOrigin - multi_lane; // the left top pixel of the left block
	for (nni = 0; nni < uiHeight - 1; nni++) 
	{
		for (nnj = 0; nnj < multi_lane; nnj++)
		{
			multiRec[nni * multi_lane + nnj] = *(nnRoiTemp + nni*iPicStride + nnj);
		}
	}

	//fill the top
	nnRoiTemp = piRoiOrigin - multi_lane - iPicStride; // the left bottom pixel of the left top block
	for (nni = 0; nni < (uiWidth - 1 + multi_lane); nni++) 
	{
		for (nnj = 0; nnj < multi_lane; nnj++)
		{
			multiRec[(multi_lane * (uiHeight - 1)) + nni*multi_lane + nnj] = *(nnRoiTemp + nni - nnj*iPicStride);
		}
	}

  }
  else // reference samples are partially available
  {
    // all above units have "unitWidth" samples each, all left/below-left units have "unitHeight" samples each
    const Int  iTotalSamples = (iLeftUnits * unitHeight) + ((iAboveUnits + 1) * unitWidth);
    Pel  piIntraLine[5 * MAX_CU_SIZE];
    Pel  *piIntraLineTemp;
    const Bool *pbNeighborFlags;

    // Initialize
    for (i=0; i<iTotalSamples; i++)
    {
      piIntraLine[i] = iDCValue;
    }
	
	//Pel nnIntraLine[ (MAX_CU_SIZE*2+multi_lane)*multi_lane /* top 3 */ + MAX_CU_SIZE*2*multi_lane /* left 2*/ ];
	for (i = 0; i < nnTotalPixels; i++)
	{
		//nnIntraLine[i] = iDCValue;
	}
	
    // Fill top-left sample
    piRoiTemp = piRoiOrigin - iPicStride - 1;
    piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight); //pixel-wise
    pbNeighborFlags = bNeighborFlags + iLeftUnits; //basic-unit-wise

	nnRoiTemp = piRoiOrigin - multi_lane - iPicStride; // the left bottom pixel of the left top block

    if (*pbNeighborFlags)
    {
#if O0043_BEST_EFFORT_DECODING
      Pel topLeftVal=piRoiTemp[0] << bitDepthDelta;
#else
      Pel topLeftVal=piRoiTemp[0];
#endif
      for (i=0; i<unitWidth; i++)
      {
        piIntraLineTemp[i] = topLeftVal;
      }

	  for (nni = 0; nni < multi_lane; nni++) //horizontal movement
	  {
		  for (nnj = 0; nnj < multi_lane; nnj++) //vertial movement
		  {
			  multiRec[(multi_lane * (uiHeight - 1)) + nni* multi_lane + nnj] = *(nnRoiTemp + nni - nnj*iPicStride);
		  }
	  }

    }

    // Fill left & below-left samples (downwards)
    piRoiTemp += iPicStride;
    piIntraLineTemp--;
    pbNeighborFlags--;

	nnRoiTemp = piRoiOrigin - multi_lane; // the left top pixel of the left bottom block

    for (j=0; j<iLeftUnits; j++) //downwards
    {
      if (*pbNeighborFlags)
      {
        for (i=0; i<unitHeight; i++)
        {
#if O0043_BEST_EFFORT_DECODING
          piIntraLineTemp[-i] = piRoiTemp[i*iPicStride] << bitDepthDelta;
#else
          piIntraLineTemp[-i] = piRoiTemp[i*iPicStride];
#endif
	
		  int nnHeightTmp = j*unitHeight + i;
		  for (nnj = 0; nnj < multi_lane; nnj++)
		  {
			  multiRec[nnHeightTmp * multi_lane + nnj] = *(nnRoiTemp + nnHeightTmp*iPicStride + nnj);
		  }
        }
      }
      piRoiTemp += unitHeight*iPicStride;
      piIntraLineTemp -= unitHeight;
      pbNeighborFlags--;
	
    }

    // Fill above & above-right samples (left-to-right) (each unit has "unitWidth" samples)
    piRoiTemp = piRoiOrigin - iPicStride;
    // offset line buffer by iNumUints2*unitHeight (for left/below-left) + unitWidth (for above-left)
    piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight) + unitWidth;
    pbNeighborFlags = bNeighborFlags + iLeftUnits + 1;

	nnRoiTemp = piRoiOrigin - multi_lane - iPicStride; // the left bottom pixel of the left top block

    for (j=0; j<iAboveUnits; j++)
    {
      if (*pbNeighborFlags)
      {
        for (i=0; i<unitWidth; i++)
        {
#if O0043_BEST_EFFORT_DECODING
          piIntraLineTemp[i] = piRoiTemp[i] << bitDepthDelta;
#else
          piIntraLineTemp[i] = piRoiTemp[i];
#endif
		  
		  int nnWidthTmp = multi_lane + (j*unitWidth + i);
		  for (nnj = 0; nnj < multi_lane; nnj++)
		  {
			  multiRec[(multi_lane * (uiHeight - 1)) + nnWidthTmp*multi_lane + nnj] = *(nnRoiTemp + nnWidthTmp - nnj*iPicStride);
		  }
         
        }
      }
      piRoiTemp += unitWidth;
      piIntraLineTemp += unitWidth;
      pbNeighborFlags++;
    }

    // Pad reference samples when necessary
    Int iCurrJnit = 0;
    Pel  *piIntraLineCur   = piIntraLine;
    const UInt piIntraLineTopRowOffset = iLeftUnits * (unitHeight - unitWidth);

    if (!bNeighborFlags[0])
    {
      // very bottom unit of bottom-left; at least one unit will be valid.
      {
        Int   iNext = 1;
        while (iNext < iTotalUnits && !bNeighborFlags[iNext]) // start from bottom-left, clockwisely find the first effective reference (rec.) sample
        {
          iNext++;
        }
        Pel *piIntraLineNext = piIntraLine + ((iNext < iLeftUnits) ? (iNext * unitHeight) : (piIntraLineTopRowOffset + (iNext * unitWidth)));

		//if (nnBestModeChromaFlag)
			//cout << iNext << " " << iLeftUnits <<  endl;

        const Pel refSample = *piIntraLineNext;

		Pel nnRefSample[MAX_CU_SIZE]; //for simplicity, use the same pixel to pad multi_lane pixels
		for (nni = 0; nni < multi_lane; nni++)
		{
			if (true) // true for dec, nnBestModeFlag for enc, to be fixed
			{
				if (iNext == ((multi_lane/unitWidth)*2+1) )
				{
					nnRefSample[nni] = *(piRoiOrigin - iPicStride*(nni+1) ); //nnRefSample[0]: the left bottom pixel of above block
				}
				else if (iNext == (multi_lane/unitWidth) )
				{
					nnRefSample[nni] = *(piRoiOrigin - multi_lane + ((uiHeight - 1) / 2 -1)*iPicStride + nni); //nnRefSample[0]: the left bottom pixel of left block
					//nnRefSample[nni] = *piIntraLineNext;
				}
				else
				{
					//cout << "my understanding is not correct" << endl;
					//cout << iNext << " " << iLeftUnits << endl;
					//assert(0);
				}
			}

			//nnRefSample[nni] = *piIntraLineNext;
		}

        // Pad unavailable samples with new value
        Int iNextOrTop = std::min<Int>(iNext, iLeftUnits); //if iNext <= iLeftUnits, only pad left unit; otherwise, pad both left and top unit
        // fill left column
        while (iCurrJnit < iNextOrTop) //iCurrJnit is unit-wise
        {
          for (i=0; i<unitHeight; i++)
          {
            piIntraLineCur[i] = refSample;
          }
          piIntraLineCur += unitHeight;
          iCurrJnit++;
        }
        // fill top row
        while (iCurrJnit < iNext)
        {
          for (i=0; i<unitWidth; i++)
          {
            piIntraLineCur[i] = refSample;
          }
          piIntraLineCur += unitWidth;
          iCurrJnit++;
        }

		//nntra
		int nniCurrJnit = 0;
		while (nniCurrJnit < iNextOrTop) // pad left
		{
			for (nni = 0; nni < unitHeight; nni++)
			{
				int nnHeightTmp = uiHeight - 2 - (nniCurrJnit * unitHeight + nni); //vertical movement
				for (nnj = 0; nnj < multi_lane; nnj++)
				{
					if (iNext == (multi_lane / unitWidth) )
					{
						multiRec[nnHeightTmp * multi_lane + nnj] = nnRefSample[nnj];
					}
					else if (iNext == ((multi_lane / unitWidth) * 2 + 1) )
					{
						multiRec[nnHeightTmp * multi_lane + nnj] = nnRefSample[0];
					}
					
				}
			}
			nniCurrJnit++;
		}
		while (nniCurrJnit < iNext)
		{
			if (nniCurrJnit == iLeftUnits) // pad top left 
			{
				for (nni = 0; nni < multi_lane; nni++)
				{
					for (nnj = 0; nnj < multi_lane; nnj++)
					{
						multiRec[(multi_lane * (uiHeight - 1)) + nni*multi_lane + nnj] = nnRefSample[nnj];
					}
				}
			}
			else // pad top above and top right
			{
				for (nni = 0; nni < unitWidth; nni++)
				{
					int nnWidthTmp = multi_lane + (nniCurrJnit - (iLeftUnits+1))*unitWidth + nni; //hortizontal movement
					for (nnj = 0; nnj < multi_lane; nnj++)
					{
						multiRec[(multi_lane * (uiHeight - 1)) + nnWidthTmp*multi_lane + nnj] = nnRefSample[nnj];
					}
				}
			}
			nniCurrJnit++;
		}
		assert(iCurrJnit == nniCurrJnit);
      }
    }

    // pad all other reference samples.
    while (iCurrJnit < iTotalUnits)
    {
      if (!bNeighborFlags[iCurrJnit]) // samples not available
      {
        {
          const Int numSamplesInCurrUnit = (iCurrJnit >= iLeftUnits) ? unitWidth : unitHeight;
          const Pel refSample = *(piIntraLineCur-1); //use the left effective reference(rec.) sample to pad
          for (i=0; i<numSamplesInCurrUnit; i++)
          {
            piIntraLineCur[i] = refSample;
          }
          piIntraLineCur += numSamplesInCurrUnit;
          
		  //nntra
		  if (iCurrJnit < iLeftUnits)
		  {
			  for (nni = 0; nni < unitHeight; nni++)
			  {
				  int nnHeightTmp = uiHeight - 2 - (iCurrJnit * unitHeight + nni); //vertical movement
				  for (nnj = 0; nnj < multi_lane; nnj++)
				  {
					  multiRec[nnHeightTmp * multi_lane + nnj] = multiRec[(nnHeightTmp + unitHeight) * multi_lane + nnj];
				  }
			  }
		  }
		  else if (iCurrJnit == iLeftUnits)
		  {
			  for (nni = 0; nni < multi_lane; nni++)
			  {
				  for (nnj = 0; nnj < multi_lane; nnj++)
				  {
					  multiRec[(multi_lane * (uiHeight - 1)) + nni*multi_lane + nnj] = multiRec[ nni ];
				  }
			  }
		  }
		  else
		  {
			  for (nni = 0; nni < unitWidth; nni++)
			  {
				  int nnWidthTmp = multi_lane + (iCurrJnit - (iLeftUnits + 1))*unitWidth + nni; //hortizontal movement 
				  for (nnj = 0; nnj < multi_lane; nnj++)
				  {
					  multiRec[(multi_lane * (uiHeight - 1)) + nnWidthTmp*multi_lane + nnj] = \
						  multiRec[(multi_lane * (uiHeight - 1)) + (nnWidthTmp - 1)*multi_lane + nnj];
				  }
			  }
		  }
		  
		  
		  
		  iCurrJnit++;
        }
      }
      else
      {
        piIntraLineCur += (iCurrJnit >= iLeftUnits) ? unitWidth : unitHeight;
        iCurrJnit++;
      }
    }

    // Copy processed samples

    piIntraLineTemp = piIntraLine + uiHeight + unitWidth - 2; //piIntraLineTemp, used as tmp pointer
    // top left, top and top right samples
	// for piIntraTemp, left and left-bottom consumes uiHeight-1 pixels, left-top consumes unitWidth pixels, top and top-right consumes uiWidth-1 pixels,
	// so (uiHeight-1)+(unitWidth-1) = uiHeight + unitWidth - 2
    for (i=0; i<uiWidth; i++)
    {
      piIntraTemp[i] = piIntraLineTemp[i];
    }

    piIntraLineTemp = piIntraLine + uiHeight - 1;
    for (i=1; i<uiHeight; i++)
    {
      piIntraTemp[i*uiWidth] = piIntraLineTemp[-i];
    }
  }
}

Bool TComPrediction::filteringIntraReferenceSamples(const ComponentID compID, UInt uiDirMode, UInt uiTuChWidth, UInt uiTuChHeight, const ChromaFormat chFmt, const Bool intraReferenceSmoothingDisabled)
{
  Bool bFilter;

  if (!filterIntraReferenceSamples(toChannelType(compID), chFmt, intraReferenceSmoothingDisabled))
  {
    bFilter=false;
  }
  else
  {
    assert(uiTuChWidth>=4 && uiTuChHeight>=4 && uiTuChWidth<128 && uiTuChHeight<128);

    if (uiDirMode == DC_IDX)
    {
      bFilter=false; //no smoothing for DC or LM chroma
    }
    else
    {
      Int diff = min<Int>(abs((Int) uiDirMode - HOR_IDX), abs((Int)uiDirMode - VER_IDX));
      UInt sizeIndex=g_aucConvertToBit[uiTuChWidth];
      assert(sizeIndex < MAX_INTRA_FILTER_DEPTHS);
      bFilter = diff > m_aucIntraFilter[toChannelType(compID)][sizeIndex];
    }
  }
  return bFilter;
}

Bool isAboveLeftAvailable( const TComDataCU* pcCU, UInt uiPartIdxLT )
{
  Bool bAboveLeftFlag;
  UInt uiPartAboveLeft;
  const TComDataCU* pcCUAboveLeft = pcCU->getPUAboveLeft( uiPartAboveLeft, uiPartIdxLT );
  if(pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
  {
    bAboveLeftFlag = ( pcCUAboveLeft && pcCUAboveLeft->isIntra( uiPartAboveLeft ) );
  }
  else
  {
    bAboveLeftFlag = (pcCUAboveLeft ? true : false);
  }
  return bAboveLeftFlag;
}

Int isAboveAvailable( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxRT, Bool *bValidFlags )
{
  const UInt uiRasterPartBegin = g_auiZscanToRaster[uiPartIdxLT];
  const UInt uiRasterPartEnd = g_auiZscanToRaster[uiPartIdxRT]+1;
  const UInt uiIdxStep = 1;
  Bool *pbValidFlags = bValidFlags;
  Int iNumIntra = 0;

  for ( UInt uiRasterPart = uiRasterPartBegin; uiRasterPart < uiRasterPartEnd; uiRasterPart += uiIdxStep )
  {
    UInt uiPartAbove;
    const TComDataCU* pcCUAbove = pcCU->getPUAbove( uiPartAbove, g_auiRasterToZscan[uiRasterPart] );
    if(pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
    {
      if ( pcCUAbove && pcCUAbove->isIntra( uiPartAbove ) )
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    else
    {
      if (pcCUAbove)
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    pbValidFlags++;
  }
  return iNumIntra;
}

Int isLeftAvailable( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxLB, Bool *bValidFlags )
{
  const UInt uiRasterPartBegin = g_auiZscanToRaster[uiPartIdxLT];
  const UInt uiRasterPartEnd = g_auiZscanToRaster[uiPartIdxLB]+1;
  const UInt uiIdxStep = pcCU->getPic()->getNumPartInCtuWidth();
  Bool *pbValidFlags = bValidFlags;
  Int iNumIntra = 0;

  for ( UInt uiRasterPart = uiRasterPartBegin; uiRasterPart < uiRasterPartEnd; uiRasterPart += uiIdxStep )
  {
    UInt uiPartLeft;
    const TComDataCU* pcCULeft = pcCU->getPULeft( uiPartLeft, g_auiRasterToZscan[uiRasterPart] );
    if(pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
    {
      if ( pcCULeft && pcCULeft->isIntra( uiPartLeft ) )
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    else
    {
      if ( pcCULeft )
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    pbValidFlags--; // opposite direction
  }

  return iNumIntra;
}

Int isAboveRightAvailable( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxRT, Bool *bValidFlags )
{
  const UInt uiNumUnitsInPU = g_auiZscanToRaster[uiPartIdxRT] - g_auiZscanToRaster[uiPartIdxLT] + 1;
  Bool *pbValidFlags = bValidFlags;
  Int iNumIntra = 0;

  for ( UInt uiOffset = 1; uiOffset <= uiNumUnitsInPU; uiOffset++ )
  {
    UInt uiPartAboveRight;
    const TComDataCU* pcCUAboveRight = pcCU->getPUAboveRight( uiPartAboveRight, uiPartIdxRT, uiOffset );
    if(pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
    {
      if ( pcCUAboveRight && pcCUAboveRight->isIntra( uiPartAboveRight ) )
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    else
    {
      if ( pcCUAboveRight )
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    pbValidFlags++;
  }

  return iNumIntra;
}

Int isBelowLeftAvailable( const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxLB, Bool *bValidFlags )
{
  const UInt uiNumUnitsInPU = (g_auiZscanToRaster[uiPartIdxLB] - g_auiZscanToRaster[uiPartIdxLT]) / pcCU->getPic()->getNumPartInCtuWidth() + 1;
  Bool *pbValidFlags = bValidFlags;
  Int iNumIntra = 0;

  for ( UInt uiOffset = 1; uiOffset <= uiNumUnitsInPU; uiOffset++ )
  {
    UInt uiPartBelowLeft;
    const TComDataCU* pcCUBelowLeft = pcCU->getPUBelowLeft( uiPartBelowLeft, uiPartIdxLB, uiOffset );
    if(pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
    {
      if ( pcCUBelowLeft && pcCUBelowLeft->isIntra( uiPartBelowLeft ) )
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    else
    {
      if ( pcCUBelowLeft )
      {
        iNumIntra++;
        *pbValidFlags = true;
      }
      else
      {
        *pbValidFlags = false;
      }
    }
    pbValidFlags--; // opposite direction
  }

  return iNumIntra;
}
//! \}
