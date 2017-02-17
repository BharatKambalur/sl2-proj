# sl2-proj

Classes:ID-R-G-B Values :
LaneMarking:-1<br />
Misc:0-0-0-0<br />
Sky:1<br />
Road:3<br />
Sidewalk:4<br />
Vegetation:6<br />

Variable Names:

RGB Image -> im <br />
SLIC Output -> segments<br />
GT Text Array -> gt<br />
Gray Image -> im_g<br />
BigX -> Observation matrix with all training examples and corresponding features<br />
BigY -> Matrix with class labels<br />

Create a folder 'dataset' in the root folder.<br />
Create the structure:<br />  dataset<br />
                           -> SYNTHIA_RAND_CVPR16<br />
                                 -> SLIC<\br><br />
To generate SLIC .npy files, run utils -> SuperPixel_Batch.py<br />
Please set variables list_start and list_end<br /><br />

Dataset Distribution:<br />
Bharat: 1000 - 3999<br />
Tharun: 4000 - 10999<br />
Girish: 11000 - 13406<br />

File Names and corresponding numbers:
0 : ap_000_01-11-2015_19-20-57_000000_0_Rand_0.png
999 : ap_000_01-11-2015_19-20-57_000025_0_Rand_9.png
1000 : ap_000_01-11-2015_19-20-57_000025_1_Rand_0.png
1999 : ap_000_02-11-2015_08-27-21_000016_1_Rand_6.png
2000 : ap_000_02-11-2015_08-27-21_000016_1_Rand_7.png
2999 : ap_000_02-11-2015_08-27-21_000043_1_Rand_5.png
3000 : ap_000_02-11-2015_08-27-21_000043_1_Rand_6.png
3999 : ap_000_02-11-2015_08-27-21_000068_2_Rand_2.png
4000 : ap_000_02-11-2015_08-27-21_000068_2_Rand_3.png
4999 : ap_000_02-11-2015_08-27-21_000094_0_Rand_2.png
5000 : ap_000_02-11-2015_08-27-21_000094_0_Rand_3.png
5999 : ap_000_02-11-2015_08-27-21_000119_1_Rand_3.png
6000 : ap_000_02-11-2015_08-27-21_000119_1_Rand_4.png
6999 : ap_000_02-11-2015_18-02-19_000017_0_Rand_10.png
7000 : ap_000_02-11-2015_18-02-19_000017_0_Rand_11.png
7999 : ap_000_02-11-2015_18-02-19_000042_2_Rand_13.png
8000 : ap_000_02-11-2015_18-02-19_000042_2_Rand_14.png
8999 : ap_000_02-11-2015_18-02-19_000065_3_Rand_2.png
9000 : ap_000_02-11-2015_18-02-19_000065_3_Rand_3.png
9999 : ap_000_02-11-2015_18-02-19_000087_1_Rand_2.png
10000 : ap_000_02-11-2015_18-02-19_000087_1_Rand_3.png
10999 : ap_000_02-11-2015_18-02-19_000110_0_Rand_7.png
11000 : ap_000_02-11-2015_18-02-19_000110_0_Rand_8.png
11999 : ap_000_02-11-2015_18-02-19_000132_0_Rand_10.png
12000 : ap_000_02-11-2015_18-02-19_000132_0_Rand_11.png
12999 : ap_000_02-11-2015_18-02-19_000153_2_Rand_0.png
13000 : ap_000_02-11-2015_18-02-19_000153_2_Rand_1.png
13406 : ap_000_02-11-2015_18-02-19_000162_3_Rand_4.png
