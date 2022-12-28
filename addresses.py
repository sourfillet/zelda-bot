from collections import OrderedDict

#link
link_x =				0x0070
link_y =				0x0084
link_raft_ladder_x =	0x007B
link_raft_ladder_y =	0x008F
link_sword_x =			0x007D
link_sword_y =			0x0091
link_bolt_x =			0x007E
link_bolt_y =			0x0092
link_boomerang_bait_x = 0x007F
link_boomerang_bait_y = 0x0093
link_bomb_fire1_x = 	0x0080
link_bomb_fire1_y = 	0x0094
link_fire2_x = 			0x0081
link_fire2_y = 			0x0095
link_arrow_rod_x = 		0x0082
link_arrow_rod_y = 		0x0096
link_item_x_pos = 		0x0083
link_item_y_pos = 		0x0097
link_status = 			0x00AC
link_delta = 			0x0394
link_movement = 		0x03A8
link_direction = 		0x03F8 #1r, 2l, 4d, 8u
link_rupees = 			0x066D
link_keys = 			0x066E
link_heart_containers = 0x066F
link_hit_points = 		0x0670
link_bombs = 			0x0658
link_current_item = 	0x0656
link_address = OrderedDict()
link_address["Link X"] = link_x
link_address["Link Y"] = link_y
link_address["Sword X"] = link_sword_x
link_address["Sword Y"] = link_sword_y
link_address["Current item"] = link_current_item
link_address["Delta"] = link_delta
link_address["Status"] = link_status
link_address["Rupees"] = link_rupees
link_address["Keys"] = link_keys
link_address["Heart Containers"] = link_heart_containers
link_address["Movement"] = link_movement
link_address["Direction"] = link_direction

#inventory
inv_triforce_pieces = 0x0671
inv_current_sword = 0x0657
inv_arrow_status = 0x0659
inv_has_bow = 0x065A
inv_candle_status = 0x065B
inv_has_whistle = 0x065C
inv_has_food = 0x065D
inv_compass = 0x0667
inv_map = 0x0668
inv_has_clock = 0x066C
inv_has_boom = 0x0674

#gannon
gannon_stun_duration = 0x0029
gannon_arrow_vul_dur = 0x00AD

#map
map_current_level = 0x0010
map_loading_ns = 0x0058
map_drawing_flag = 0x00E3
#map_changing = 0x00E6
map_dir_change = 0x00E7	#1r, 2l, 4d, 8u
#map_scroll_lr = 0x00E8
map_scroll_lr2 = 0x00FD
#map_scroll_ud = 0x00E9
map_scroll_ud2 = 0x00ED
map_current_room = 0x00EB
map_destination = 0x00EC
map_dir_entered = 0x00EE

#nums
num_enemies_screen_killed = 0x034F
num_enemies_screen_max  = 0x034E
num_rupees_added = 0x067D
num_rupees_removed = 0x067E

#backend stuff
backend_mode = 0x0012	#6->prep scroll, 7->scrolling, 4->finishing scroll
backend_tenth_bomb = 0x0050 #resets to 0 when it reaches 10, enemy will drop bomb
backend_frame_count = 0x0015

#controller
controller_1_press = 0x00F8
controller_2_press = 0x00F9
controller_1_down = 0x00FA
controller_2_down = 0x00FB

#validate
VAL_num_sprites = 0x0059
VAL_right_screen_wipe = 0x007D


"""
NOTE: Not currently working. Addresses seem to be wrong.
"""
#enemies
#enemy_tally = 0x0050	????????????????????
enemy_x_y = [
	#enemies:
	#  x	  y	  direction
	[0x0071,0x0085,0x03F9], #1
	[0x0072,0x0086,0x03FA], #2
	[0x0073,0x0087,0x03FB], #3
	[0x0074,0x0088,0x03FC], #4
	[0x0075,0x0089,0x03FD], #5  
	[0x0076,0x008A,0x03FE], #6
	[0x0077,0x008B,0x03FF], #7
	[0x0078,0x008C,0x0400], #8
	[0x0079,0x008D,0x0401], #9 
	[0x007A,0x008E,0x0402], #10
	[0x007B,0x008F,0x0403], #11
	]

enemy_projectile_exists = [
    [0x00B4, 0x00B5, 0x00B6, 0x00B3, 0x00B7]
    ]

enemy_projectile_x_y = [
    #   x     y     direction
    [0x0078,0x008C,0x00A0],   #1
    [0x0079,0x008D,0x00A1],   #2
    [0x007A,0x008E,0x00A2],   #3
    [0x0077,0x008B,0x009F],   #4
    ]