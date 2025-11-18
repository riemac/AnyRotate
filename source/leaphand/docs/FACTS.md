# 事实记录
这里记录一些关于Leaphand的事实和数据，以供参考。

## Leaphand 关节索引和名称

1. LeapHand机器人的刚体连杆组成部分:
- 手掌: palm_lower
- 食指: mcp_joint -> pip -> dip -> fingertip -> index_tip_head 
- 拇指: thumb_temp_base -> thumb_pip -> thumb_dip -> thumb_fingertip -> thumb_tip_head
- 中指: mcp_joint_2 -> pip_2 -> dip_2 -> fingertip_2 -> middle_tip_head
- 无名指: mcp_joint_3 -> pip_3 -> dip_3 -> fingertip_3 -> ring_tip_head
  
2. 关节索引：

- joints = [a_1, a_12, a_5, a_9, a_0, a_13, a_4, a_8, a_2, a_14, a_6, a_10, a_3, a_15, a_7, a_11]
- index finger: a_o~a_3
- middle finger: a_4~a_7
- little finger: a_8~a_11
- thumb: a_12~a_15

3. 关节限位：