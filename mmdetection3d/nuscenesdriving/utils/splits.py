# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Dict, List
import json

from nuscenesdriving import NuScenesDrivIng


driving_train = [f'{i:03d}' for i in range(50)]
driving_val = [f'{i:03d}' for i in range(50, 100)]
driving_test = [f'{i:03d}' for i in range(100, 150)]

carla_train = [
        "drivIng_long_0_expert_traj_route20_11_08_20_50_08",
        "drivIng_long_8_expert_traj_route2_10_13_19_06_49",
        "drivIng_long_4_expert_traj_route17_10_20_10_38_21",
        "drivIng_long_8_expert_traj_route12_10_14_16_03_12",
        "drivIng_long_2_expert_traj_route18_11_10_08_26_28",
        "drivIng_long_6_expert_traj_route15_11_09_09_27_39",
        "drivIng_long_3_expert_traj_route3_10_13_20_12_08",
        "drivIng_long_6_expert_traj_route1_11_09_13_30_31",
        "drivIng_long_8_expert_traj_route4_10_14_14_42_31",
        "drivIng_long_5_expert_traj_route5_10_15_14_32_15",
        "drivIng_long_4_expert_traj_route29_10_20_13_42_42",
        "drivIng_long_3_expert_traj_route6_10_22_00_09_52",
        "drivIng_long_8_expert_traj_route4_10_13_19_15_34",
        "drivIng_long_0_expert_traj_route1_11_09_13_27_42",
        "drivIng_long_5_expert_traj_route39_10_21_14_09_06",
        "drivIng_long_1_expert_traj_route6_11_08_06_25_29",
        "drivIng_long_1_expert_traj_route8_11_08_07_12_40",
        "drivIng_long_0_expert_traj_route5_11_08_01_09_36",
        "drivIng_long_4_expert_traj_route8_10_15_00_35_23",
        "drivIng_long_8_expert_traj_route7_10_20_20_05_50",
        "drivIng_long_7_expert_traj_route2_11_08_01_52_00",
        "drivIng_long_2_expert_traj_route11_11_09_17_29_28",
        "drivIng_long_2_expert_traj_route3_11_09_15_29_26",
        "drivIng_long_8_expert_traj_route1_10_20_18_20_13",
        "drivIng_long_6_expert_traj_route6_11_09_20_14_55",
        "drivIng_long_6_expert_traj_route9_11_09_05_35_40",
        "drivIng_long_9_expert_traj_route9_10_14_16_12_17",
        "drivIng_long_8_expert_traj_route5_10_13_19_19_52",
        "drivIng_long_6_expert_traj_route17_11_10_09_35_24",
        "drivIng_long_9_expert_traj_route12_10_14_18_42_42",
        "drivIng_long_0_expert_traj_route19_11_08_20_32_32",
        "drivIng_long_1_expert_traj_route3_11_08_00_34_28",
        "drivIng_long_9_expert_traj_route2_10_13_17_43_44",
        "drivIng_long_6_expert_traj_route15_11_08_08_54_11",
        "drivIng_long_2_expert_traj_route4_11_08_21_11_33",
        "drivIng_long_4_expert_traj_route11_10_15_15_53_57",
        "drivIng_long_9_expert_traj_route6_10_13_20_08_23",
        "drivIng_long_1_expert_traj_route8_11_09_21_21_21",
        "drivIng_long_9_expert_traj_route10_10_14_18_25_21",
        "drivIng_long_7_expert_traj_route5_11_09_05_47_32",
        "drivIng_long_8_expert_traj_route15_10_20_21_18_21",
        "drivIng_long_2_expert_traj_route3_11_08_21_06_36",
        "drivIng_long_9_expert_traj_route14_10_14_19_10_41",
        "drivIng_long_8_expert_traj_route14_10_14_00_47_49",
        "drivIng_long_1_expert_traj_route2_11_08_00_26_58",
        "drivIng_long_2_expert_traj_route17_11_09_05_02_47",
        "drivIng_long_1_expert_traj_route3_11_08_22_41_33",
        "drivIng_long_6_expert_traj_route8_11_09_05_06_13",
        "drivIng_long_3_expert_traj_route4_10_21_20_24_49",
        "drivIng_long_7_expert_traj_route2_11_08_23_27_33",
        "drivIng_long_2_expert_traj_route20_11_09_11_13_15",
        "drivIng_long_9_expert_traj_route2_10_21_19_58_44",
        "drivIng_long_5_expert_traj_route24_10_20_13_47_16",
        "drivIng_long_2_expert_traj_route9_11_08_01_28_27",
        "drivIng_long_0_expert_traj_route13_11_09_23_15_43",
        "drivIng_long_0_expert_traj_route5_11_09_15_59_11",
        "drivIng_long_6_expert_traj_route14_11_09_09_12_01",
        "drivIng_long_0_expert_traj_route15_11_10_03_46_14",
        "drivIng_long_7_expert_traj_route6_11_09_06_02_28",
        "drivIng_long_8_expert_traj_route2_10_14_12_21_36",
        "drivIng_long_0_expert_traj_route3_11_09_13_48_47",
        "drivIng_long_2_expert_traj_route10_11_08_01_34_34",
        "drivIng_long_8_expert_traj_route8_10_13_21_27_51",
        "drivIng_long_6_expert_traj_route10_11_09_06_01_52",
        "drivIng_long_4_expert_traj_route12_10_15_15_57_51",
        "drivIng_long_8_expert_traj_route13_10_14_16_18_32",
        "drivIng_long_5_expert_traj_route3_10_15_13_13_35",
        "drivIng_long_9_expert_traj_route11_10_14_18_31_52",
        "drivIng_long_7_expert_traj_route0_11_08_20_19_05",
        "drivIng_long_6_expert_traj_route10_11_08_05_22_34",
        "drivIng_long_5_expert_traj_route34_10_20_15_10_18",
        "drivIng_long_2_expert_traj_route0_11_09_13_06_53",
        "drivIng_long_1_expert_traj_route9_11_09_23_19_41",
        "drivIng_long_1_expert_traj_route2_11_09_15_53_42",
        "drivIng_long_9_expert_traj_route3_10_14_12_01_09",
        "drivIng_long_2_expert_traj_route12_11_09_17_39_18",
        "drivIng_long_8_expert_traj_route12_10_20_21_07_10",
        "drivIng_long_1_expert_traj_route9_11_08_07_37_26",
        "drivIng_long_5_expert_traj_route22_10_20_13_04_21",
        "drivIng_long_9_expert_traj_route7_10_13_21_04_45",
        "drivIng_long_5_expert_traj_route31_10_20_14_56_15",
        "drivIng_long_4_expert_traj_route5_10_14_20_21_17",
        "drivIng_long_2_expert_traj_route21_11_09_11_18_26",
        "drivIng_long_3_expert_traj_route3_10_21_18_28_32",
        "drivIng_long_7_expert_traj_route5_11_08_06_05_22",
        "drivIng_long_0_expert_traj_route15_11_08_08_22_48",
        "drivIng_long_0_expert_traj_route0_11_09_13_06_10",
        "drivIng_long_6_expert_traj_route18_11_10_09_44_51",
        "drivIng_long_2_expert_traj_route6_11_07_23_36_45",
        "drivIng_long_4_expert_traj_route0_10_14_11_09_09",
        "drivIng_long_2_expert_traj_route15_11_09_01_56_54",
        "drivIng_long_0_expert_traj_route7_11_08_01_40_24",
        "drivIng_long_8_expert_traj_route6_10_14_15_01_48",
        "drivIng_long_6_expert_traj_route8_11_09_20_38_14",
        "drivIng_long_1_expert_traj_route6_11_09_06_14_40",
        "drivIng_long_5_expert_traj_route32_10_20_15_00_31",
        "drivIng_long_4_expert_traj_route6_10_13_18_29_35",
        "drivIng_long_2_expert_traj_route11_11_08_01_43_11",
        "drivIng_long_1_expert_traj_route3_11_09_15_57_56",
        "drivIng_long_9_expert_traj_route1_10_21_19_35_33",
        "drivIng_long_4_expert_traj_route5_10_15_14_31_05",
        "drivIng_long_3_expert_traj_route0_10_15_11_46_43",
        "drivIng_long_8_expert_traj_route10_10_20_20_56_56",
        "drivIng_long_9_expert_traj_route8_10_14_15_35_12",
        "drivIng_long_2_expert_traj_route11_11_09_00_37_28",
        "drivIng_long_2_expert_traj_route15_11_09_20_38_38",
        "drivIng_long_8_expert_traj_route11_10_13_23_11_56",
        "drivIng_long_4_expert_traj_route2_10_13_16_19_32",
        "drivIng_long_6_expert_traj_route8_11_08_04_24_26",
        "drivIng_long_8_expert_traj_route11_10_20_21_01_57",
        "drivIng_long_1_expert_traj_route8_11_09_08_22_56",
        "drivIng_long_6_expert_traj_route16_11_10_09_28_54",
        "drivIng_long_3_expert_traj_route3_10_15_12_53_39",
        "drivIng_long_4_expert_traj_route0_10_13_16_10_37",
        "drivIng_long_5_expert_traj_route27_10_20_14_38_43",
        "drivIng_long_9_expert_traj_route2_10_20_17_31_29",
        "drivIng_long_2_expert_traj_route2_11_08_20_45_41",
        "drivIng_long_8_expert_traj_route7_10_13_20_35_37",
        "drivIng_long_4_expert_traj_route11_10_15_07_11_26",
        "drivIng_long_2_expert_traj_route8_11_08_22_38_40",
        "drivIng_long_9_expert_traj_route1_10_20_17_27_39",
        "drivIng_long_9_expert_traj_route3_10_21_22_30_59",
        "drivIng_long_8_expert_traj_route8_10_14_15_16_05",
        "drivIng_long_9_expert_traj_route4_10_13_19_17_51",
        "drivIng_long_9_expert_traj_route13_10_13_22_21_12",
        "drivIng_long_1_expert_traj_route10_11_08_07_40_48",
        "drivIng_long_2_expert_traj_route13_11_09_01_17_39",
        "drivIng_long_4_expert_traj_route7_10_14_20_52_52",
        "drivIng_long_2_expert_traj_route7_11_08_22_34_13",
        "drivIng_long_8_expert_traj_route8_10_20_20_11_33",
        "drivIng_long_0_expert_traj_route0_11_07_22_15_52",
        "drivIng_long_2_expert_traj_route1_11_07_22_25_16",
        "drivIng_long_5_expert_traj_route38_10_20_18_25_55",
        "drivIng_long_5_expert_traj_route35_10_20_15_15_38",
        "drivIng_long_9_expert_traj_route4_10_14_12_11_22",
        "drivIng_long_1_expert_traj_route1_11_07_23_04_31",
        "drivIng_long_4_expert_traj_route11_10_13_19_26_43",
        "drivIng_long_5_expert_traj_route34_10_20_17_23_32",
        "drivIng_long_4_expert_traj_route7_10_13_18_34_41",
        "drivIng_long_5_expert_traj_route29_10_20_14_48_54",
        "drivIng_long_6_expert_traj_route7_11_08_04_13_45",
        "drivIng_long_8_expert_traj_route4_10_20_19_24_58",
        "drivIng_long_4_expert_traj_route10_10_15_15_48_18",
        "drivIng_long_8_expert_traj_route18_10_22_01_32_19",
        "drivIng_long_5_expert_traj_route35_10_20_17_28_45",
        "drivIng_long_6_expert_traj_route16_11_09_09_38_57",
        "drivIng_long_2_expert_traj_route18_11_09_07_18_20",
        "drivIng_long_6_expert_traj_route6_11_08_03_57_50",
        "drivIng_long_0_expert_traj_route6_11_09_16_06_33",
        "drivIng_long_4_expert_traj_route19_10_20_11_29_49",
        "drivIng_long_8_expert_traj_route5_10_20_19_57_47",
        "drivIng_long_6_expert_traj_route7_11_09_04_06_51",
        "drivIng_long_1_expert_traj_route9_11_09_10_33_28",
        "drivIng_long_1_expert_traj_route10_11_09_10_46_36",
        "drivIng_long_2_expert_traj_route17_11_10_06_06_26",
        "drivIng_long_6_expert_traj_route14_11_08_08_30_03",
        "drivIng_long_1_expert_traj_route6_11_09_19_30_16",
        "drivIng_long_5_expert_traj_route9_10_15_15_01_51",
        "drivIng_long_3_expert_traj_route6_10_15_14_26_47",
        "drivIng_long_6_expert_traj_route3_11_09_16_47_55",
        "drivIng_long_9_expert_traj_route4_10_21_23_05_24",
        "drivIng_long_0_expert_traj_route6_11_08_01_22_24",
        "drivIng_long_9_expert_traj_route8_10_13_21_50_50",
        "drivIng_long_6_expert_traj_route15_11_10_09_23_41",
        "drivIng_long_5_expert_traj_route33_10_20_15_05_19",
        "drivIng_long_4_expert_traj_route5_10_13_18_21_50",
        "drivIng_long_4_expert_traj_route27_10_20_13_29_52",
        "drivIng_long_6_expert_traj_route9_11_08_04_51_37",
        "drivIng_long_4_expert_traj_route2_10_15_12_21_27",
        "drivIng_long_8_expert_traj_route12_10_13_23_56_50",
        "drivIng_long_9_expert_traj_route6_10_14_12_35_31",
        "drivIng_long_9_expert_traj_route2_10_14_11_55_40",
        "drivIng_long_4_expert_traj_route1_10_14_14_54_29",
        "drivIng_long_2_expert_traj_route0_11_08_20_18_46",
        "drivIng_long_9_expert_traj_route15_10_14_19_18_57",
        "drivIng_long_2_expert_traj_route0_11_07_22_16_21",
        "drivIng_long_4_expert_traj_route10_10_13_19_22_38",
        "drivIng_long_1_expert_traj_route1_11_08_21_07_30",
        "drivIng_long_2_expert_traj_route4_11_09_15_34_33",
        "drivIng_long_3_expert_traj_route4_10_13_20_21_30",
        "drivIng_long_6_expert_traj_route0_11_07_22_16_22",
        "drivIng_long_4_expert_traj_route0_10_14_11_39_46",
        "drivIng_long_9_expert_traj_route1_10_14_11_13_04",
        "drivIng_long_3_expert_traj_route0_10_21_14_08_27",
        "drivIng_long_0_expert_traj_route2_11_07_22_42_36",
        "drivIng_long_8_expert_traj_route15_10_14_01_29_36",
        "drivIng_long_4_expert_traj_route20_10_20_11_35_02",
        "drivIng_long_2_expert_traj_route7_11_07_23_47_13",
        "drivIng_long_9_expert_traj_route7_10_14_14_20_19",
        "drivIng_long_1_expert_traj_route1_11_09_15_12_34",
        "drivIng_long_6_expert_traj_route0_11_09_13_06_50",
        "drivIng_long_3_expert_traj_route4_10_15_13_23_19",
        "drivIng_long_8_expert_traj_route5_10_14_14_57_05",
        "drivIng_long_0_expert_traj_route10_11_08_06_29_46",
        "drivIng_long_9_expert_traj_route14_10_13_22_32_01",
        "drivIng_long_8_expert_traj_route7_10_14_15_07_58",
        "drivIng_long_8_expert_traj_route6_10_13_19_51_31",
        "drivIng_long_9_expert_traj_route3_10_20_17_34_57",
        "drivIng_long_6_expert_traj_route13_11_10_08_41_36",
        "drivIng_long_3_expert_traj_route0_10_13_16_10_34",
        "drivIng_long_6_expert_traj_route13_11_09_08_39_16",
        "drivIng_long_2_expert_traj_route13_11_09_18_03_59",
        "drivIng_long_2_expert_traj_route8_11_09_16_57_19",
        "drivIng_long_2_expert_traj_route4_11_07_23_10_21",
        "drivIng_long_8_expert_traj_route13_10_20_21_10_44",
        "drivIng_long_4_expert_traj_route16_10_15_17_41_22",
        "drivIng_long_8_expert_traj_route14_10_14_16_28_24",
        "drivIng_long_9_expert_traj_route11_10_13_22_07_15",
        "drivIng_long_1_expert_traj_route2_11_08_22_35_36",
        "drivIng_long_8_expert_traj_route13_10_14_00_43_30",
        "drivIng_long_9_expert_traj_route10_10_13_22_03_22",
        "drivIng_long_6_expert_traj_route9_11_09_20_56_43",
        "drivIng_long_5_expert_traj_route39_10_20_18_29_35",
        "drivIng_long_5_expert_traj_route8_10_15_14_53_30",
        "drivIng_long_0_expert_traj_route17_11_08_09_06_52",
        "drivIng_long_8_expert_traj_route2_10_20_18_27_16",
        "drivIng_long_4_expert_traj_route7_10_15_15_04_07",
        "drivIng_long_5_expert_traj_route7_10_15_14_46_54",
        "drivIng_long_6_expert_traj_route1_11_08_21_30_41",
        "drivIng_long_9_expert_traj_route13_10_14_19_00_17",
        "drivIng_long_7_expert_traj_route7_11_09_06_13_01",
        "drivIng_long_0_expert_traj_route3_11_07_22_52_45",
        "drivIng_long_4_expert_traj_route1_10_14_11_14_37",
        "drivIng_long_6_expert_traj_route13_11_08_07_41_21",
        "drivIng_long_3_expert_traj_route2_10_21_16_34_33",
        "drivIng_long_9_expert_traj_route12_10_13_22_10_53",
        "drivIng_long_6_expert_traj_route7_11_09_20_28_44",
        "drivIng_long_4_expert_traj_route1_10_15_11_51_53",
        "drivIng_long_8_expert_traj_route6_10_20_20_01_42",
        "drivIng_long_0_expert_traj_route7_11_09_16_21_32",
        "drivIng_long_8_expert_traj_route10_10_14_15_31_03",
        "drivIng_long_8_expert_traj_route11_10_14_15_49_27",
        "drivIng_long_5_expert_traj_route37_10_20_18_21_44",
        "drivIng_long_2_expert_traj_route6_11_09_16_37_14",
        "drivIng_long_4_expert_traj_route12_10_15_07_30_24",
        "drivIng_long_4_expert_traj_route28_10_20_13_34_30",
        "drivIng_long_9_expert_traj_route9_10_13_21_55_04",
        "drivIng_long_4_expert_traj_route12_10_13_20_16_20",
        "drivIng_long_2_expert_traj_route8_11_07_23_53_49",
        "drivIng_long_8_expert_traj_route14_10_20_21_14_34",
        "drivIng_long_2_expert_traj_route9_11_09_17_09_46",
        "drivIng_long_2_expert_traj_route12_11_09_00_55_27",
        "drivIng_long_3_expert_traj_route2_10_15_12_26_16",
        "drivIng_long_2_expert_traj_route19_11_09_09_00_41",
        "drivIng_long_7_expert_traj_route0_11_07_22_16_40",
        "drivIng_long_5_expert_traj_route30_10_20_14_52_13",
        "drivIng_long_0_expert_traj_route9_11_09_18_16_23",
        "drivIng_long_6_expert_traj_route14_11_10_09_08_41",
        "drivIng_long_8_expert_traj_route1_10_14_11_16_33",
        "drivIng_long_6_expert_traj_route1_11_07_22_23_31",
        "drivIng_long_4_expert_traj_route23_10_20_11_49_27",
        "drivIng_long_2_expert_traj_route1_11_08_20_27_37",
        "drivIng_long_4_expert_traj_route8_10_13_18_39_03",
        "drivIng_long_6_expert_traj_route0_11_08_20_18_46",
        "drivIng_long_4_expert_traj_route26_10_20_13_06_38",
        "drivIng_long_5_expert_traj_route2_10_15_13_07_46",
        "drivIng_long_4_expert_traj_route8_10_15_15_08_44",
        "drivIng_long_0_expert_traj_route13_11_08_07_27_56",
        "drivIng_long_6_expert_traj_route3_11_08_00_36_02",
        "drivIng_long_4_expert_traj_route1_10_13_16_16_40",
        "drivIng_long_0_expert_traj_route11_11_09_20_18_04",
        "drivIng_long_3_expert_traj_route0_10_14_11_39_25",
        "drivIng_long_4_expert_traj_route22_10_20_11_45_17",
        "drivIng_long_9_expert_traj_route4_10_20_17_38_46",
        "drivIng_long_4_expert_traj_route0_10_15_11_47_06",
        "drivIng_long_6_expert_traj_route3_11_08_23_39_55",
        "drivIng_long_5_expert_traj_route28_10_20_14_45_08",
        "drivIng_long_2_expert_traj_route2_11_09_14_38_22",
        "drivIng_long_7_expert_traj_route9_11_09_08_19_33",
        "drivIng_long_4_expert_traj_route15_10_15_17_36_12",
        "drivIng_long_2_expert_traj_route1_11_09_13_44_10",
        "drivIng_long_2_expert_traj_route10_11_09_17_18_53",
        "drivIng_long_6_expert_traj_route6_11_09_03_53_31",
        "drivIng_long_5_expert_traj_route25_10_20_13_50_26",
        "drivIng_long_4_expert_traj_route6_10_15_14_39_12",
        "drivIng_long_6_expert_traj_route10_11_09_21_48_22",
        "drivIng_long_2_expert_traj_route3_11_07_23_05_25",
        "drivIng_long_4_expert_traj_route10_10_15_06_48_36",
        "drivIng_long_0_expert_traj_route9_11_08_04_33_58",
        "drivIng_long_9_expert_traj_route1_10_13_17_01_47",
        "drivIng_long_1_expert_traj_route10_11_09_23_35_43",
        "drivIng_long_0_expert_traj_route2_11_09_13_37_25",
        "drivIng_long_2_expert_traj_route6_11_08_22_27_26",
        "drivIng_long_0_expert_traj_route18_11_08_20_18_15",
        "drivIng_long_5_expert_traj_route21_10_20_12_57_36",
        "drivIng_long_3_expert_traj_route6_10_13_21_15_26",
        "drivIng_long_8_expert_traj_route1_10_13_18_30_13",
        "drivIng_long_9_expert_traj_route3_10_13_18_32_13"
        ]
carla_val = [
        "drivIng_long_4_expert_traj_route6_10_14_20_33_19",
        "drivIng_long_0_expert_traj_route11_11_08_06_40_39",
        "drivIng_long_4_expert_traj_route21_10_20_11_39_50",
        "drivIng_long_5_expert_traj_route6_10_15_14_37_19",
        "drivIng_long_2_expert_traj_route9_11_09_00_06_22",
        "drivIng_long_2_expert_traj_route10_11_09_00_16_27",
        "drivIng_long_3_expert_traj_route0_10_13_17_55_55",
        "drivIng_long_3_expert_traj_route2_10_13_19_30_32",
        "drivIng_long_4_expert_traj_route2_10_14_15_59_58",
        "drivIng_long_2_expert_traj_route7_11_09_16_49_13",
        "drivIng_long_4_expert_traj_route16_10_20_10_34_14",
        "drivIng_long_2_expert_traj_route2_11_07_22_45_15",
        "drivIng_long_0_expert_traj_route10_11_09_19_58_13",
        "drivIng_long_8_expert_traj_route10_10_13_22_30_30",
        "drivIng_long_0_expert_traj_route1_11_07_22_32_44",
        "drivIng_long_0_expert_traj_route21_11_08_21_00_22",
        "drivIng_long_8_expert_traj_route16_10_20_21_21_33"
]
carla_test = carla_val


def create_splits_logs(split: str, nusc: 'NuScenesDrivIng') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenesDrivIng split.
    :param nusc: NuScenesDrivIng instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known NuScenesDrivIng split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenesDrivIng version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenesDrivIng version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenesDrivIng version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(dataset_type: str = "driving", verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The splits of the NuScenesDrivIng dataset.
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    # all_scenes = train + val + test
    # assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    if dataset_type == "driving":
        train = driving_train
        val = driving_val
        test = driving_test
    else:
        train = carla_train
        val = carla_val
        test = carla_test
    scene_splits = {'train': train, 'val': val, 'test': test}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
