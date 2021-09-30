# Learned-Intra-Prediction

## Overview

This repository contains the code and pre-trained models to reproduce the following paper:

**Heming Sun, Lu Yu, Jiro Katto, "Fully Neural Network Mode Based Intra Prediction of Variable Block Size," VCIP 2020 Best Paper**

The paper is available at https://arxiv.org/pdf/2108.02503.pdf

## Build

We build Tensorflow from source on Windows. The version is v1.8.

The dynamic library tensorflow.dll is in bin/vc2015/x64/Release, which is the same folder as execuable encoder and decoder.

The implementation is based on HEVC Test Model (HM) 16.9.

## Run

You can run the encoder by executing

```Bash
TAppEncoder -c encoder_intra_main_dft.cfg -c BasketballPass.cfg -q 22 -f 1 --SummaryOutFilename="psnr.txt" -b str.bin -o rec.yuv
```

After running encoder, you can find bitrate|y-psnr|u-psnr|v-psnr|yuv-psnr in psnr.txt, and encoding time in encodeTime.txt.

You can run the decoder by executing

```Bash
TAppDecoder -b str.bin -o dec_rec.yuv
```

After running decoder, you can find decoding time in decodeTime.txt.

## Notes

In the TLibCommon/TypeDef.h, in the case of nnStrengthenNnModeFlag being false, then it is original HM.

## Acknowledgements

We sincerely appreciate Dr. Jiahao Li and Dr. Yueyu Hu from Peking University for their kind helps and comments on our work.

