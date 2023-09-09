import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
from robosuite.utils.observables import Observable, sensor

#NEW --> we have to change that
DEFAULT_TWO_ARM_PEG_IN_HOLE_CONFIG = {
    'large_hole': False,
    'd_weight': 1,
    't_weight': 5,
    'cos_weight': 1,
    'scale_by_cos': True,
    'scale_by_d': True,
    'cos_tanh_mult': 3.0,
    'd_tanh_mult': 15.0,
    't_tanh_mult': 7.5,
    'limit_init_ori': True,
    'lift_pos_offset': 0.3,
}
ENV_HORIZON      = 250
ACTION_LIM       = 30
ACTION_DIM       = 12
AGENT_HORIZON    = int(ENV_HORIZON / (ACTION_LIM / 2))  # This is necessarily an approximation
NUM_VEC_ENVS     = 4 #changed here
PRIMITIVE        = 't'
QUAT_ANGLES_PEG  = np.array([0.5, 0.5, 0.5, -0.5])
QUAT_ANGLES_HOLE = np.array([0, -0.7071, 0.7071, 0])
BBOX_PEG         = np.array([[-0.2, +0.2], [-0.4, -0.2], [+1.5, +1.9]])
BBOX_HOLE        = np.array([[-0.2, +0.2], [+0.2, +0.4], [+1.3, +1.7]])
MIN_BBOX_PEG     = BBOX_PEG[:, 0]
MAX_BBOX_PEG     = BBOX_PEG[:, 1]
MIN_BBOX_HOLE    = BBOX_HOLE[:, 0]
MAX_BBOX_HOLE    = BBOX_HOLE[:, 1]


class TwoArmPegInHole(TwoArmEnv):
    """
    This class corresponds to the peg-in-hole task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" corresponds to either "bimanual" if a bimanual robot is used or "single-arm-opposed" if two
        single-arm robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate gripper models from gripper factory.
            For this environment, setting a value other than the default (None) will raise an AssertionError, as
            this environment is not meant to be used with any gripper at all.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool or list of bool): if True, every observation for a specific robot includes a rendered
        image. Should either be single bool if camera obs value is to be used for all
            robots or else it should be a list of the same length as "robots" param

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        peg_radius (2-tuple): low and high limits of the (uniformly sampled)
            radius of the peg

        peg_length (float): length of the peg

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Gripper specified]
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
        initialization_noise="default",
        #this is adapted from lift, we have to fix it for two arms
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        table_offset=(0, 0, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        peg_radius=(0.015, 0.03),
        peg_length=0.13,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        #camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        #renderer_config=None,
        #New
        task_config=None,
        #skill_config=None,
        skill_config_peg=None,
        skill_config_hole=None,
    ):
        # Assert that the gripper type is None
        assert gripper_types is None, "Tried to specify gripper other than None in TwoArmPegInHole environment!"

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length

        #New
        # Get config
        self.task_config = DEFAULT_TWO_ARM_PEG_IN_HOLE_CONFIG.copy()
        if task_config is not None:
            self.task_config.update(task_config)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            #camera_segmentations=camera_segmentations,
            #renderer=renderer,
            #renderer_config=renderer_config,
            #skill_config=skill_config,
            skill_config_peg=skill_config_peg,
            skill_config_hole = skill_config_hole,
        )

    '''
    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 5.0 is provided if the peg is inside the plate's hole
              - Note that we enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arms to approach each other
            - Perpendicular Distance: in [0,1], to encourage the arms to approach each other
            - Parallel Distance: in [0,1], to encourage the arms to approach each other
            - Alignment: in [0, 1], to encourage having the right orientation between the peg and hole.
            - Placement: in {0, 1}, nonzero if the peg is in the hole with a relatively correct alignment

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale

        """
        reward = 0

        # Right location and angle
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            # Grab relevant values
            t, d, cos = self._compute_orientation()
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, scale sparse reward so that the max reward is identical to its dense version
        else:
            reward *= 5.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 5.0

        return reward
    '''

    #New
    def _calculate_angle_magnitude(self, position, quat_angles, min_bbox, max_bbox):
        """Calculate angle magnitude based on position"""
        if np.all(min_bbox <= position) and np.all(position <= max_bbox):
            return np.linalg.norm(quat_angles - position) / len(quat_angles)
        else:
            return 0.5

    def reward(self, action):
        #1. stay with this primitives
        #2. maybe add / exchange some of these primitives
        reward = 0

        # Right location and angle

        r_align, r_insert, r_reach = self.staged_rewards() #r_grasp, 
        if self.reward_shaping:
            reward = max(  r_align, r_reach) #r_grasp, r_reach,
        else:
            reward = r_insert

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def staged_rewards(self):
        reach_mult = 0.10
        grasp_mult = 0.15
        align_mult = 0.85

        #r_grasp = 0
        r_reach = 0 
        r_align = 0
        reward = 0

        t, d, cos = self._compute_orientation()

        '''
        # reaching reward
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
        dist = np.linalg.norm(gripper_site_pos - hole_pos)
        #Original one for reach
        r_reach = 1 - np.tanh(1.0 * dist) #* reach_mult
        #Maple one for reach
        #r_reach = (1 - np.tan(10.0 * dist)) #* reach_mult
        
        # Orientation reward
        #one arm peg in hole  reward from maple        
        d_w = self.task_config['d_weight']
        t_w = self.task_config['t_weight']
        cos_w = self.task_config['cos_weight']

        cos_tanh_mult = self.task_config['cos_tanh_mult']
        d_tanh_mult = self.task_config['d_tanh_mult']
        t_tanh_mult = self.task_config['t_tanh_mult']

        cos_dist = 1 - (cos + 1) / 2
        cos_rew = 1 - np.tanh(cos_tanh_mult * cos_dist)
        d_rew = 1 - np.tanh(d_tanh_mult * d)
        t_rew = 1 - np.tanh(t_tanh_mult * np.abs(t))

        scale_by_cos = self.task_config['scale_by_cos']
        scale_by_d = self.task_config['scale_by_d']
        if scale_by_cos:
            t_rew *= cos_rew
        if scale_by_d:
            t_rew *= d_rew
        if scale_by_cos:
            d_rew *= cos_rew

        r_align_sum = (cos_w * cos_rew) + (d_w * d_rew) + (t_w * t_rew)
        r_align_norm = r_align_sum / float(d_w + t_w + cos_w) # normalize sum to be between 0 and 1

        r_align = r_align_norm * (align_mult)
        

        
        
        #we also have to change this for the two arm problem
        #we maybe can get inspired from the last years group
        # Orientation reward from original implementation --> we maybe have to change that
        r_align += 1 - np.tanh(d)
        r_align += 1 - np.tanh(np.abs(t))
        r_align += cos
        r_align = r_align * align_mult
        """
        #align --> last year
        """Calculate reward for 'align' primitive"""
        angle_mag_peg = self._calculate_angle_magnitude(gripper_site_pos, QUAT_ANGLES_PEG, MIN_BBOX_PEG, MAX_BBOX_PEG)
        print(angle_mag_peg)
        angle_mag_hole = self._calculate_angle_magnitude(hole_pos, QUAT_ANGLES_HOLE, MIN_BBOX_HOLE, MAX_BBOX_HOLE)
        r_align = 1 - (angle_mag_peg + angle_mag_hole)
        print(r_align)
        
        # Grab relevant values
        t, d, cos = self._compute_orientation()
            # reaching reward
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
        dist = np.linalg.norm(gripper_site_pos - hole_pos)
        reaching_reward = 1 - np.tanh(1.0 * dist)
        reward += reaching_reward

            # Orientation reward
        reward += 1 - np.tanh(d)
        reward += 1 - np.tanh(np.abs(t))
        reward += cos

        r_align = reward
        '''
        # Grab relevant values
        t, d, cos = self._compute_orientation()
            # reaching reward
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
        dist = np.linalg.norm(gripper_site_pos - hole_pos)
        reaching_reward = 1 - np.tanh(1.0 * dist)
        r_reach += reaching_reward * 0.1

        #angle_mag_peg = self._calculate_angle_magnitude(gripper_site_pos, QUAT_ANGLES_PEG, MIN_BBOX_PEG, MAX_BBOX_PEG)

        #angle_mag_hole = self._calculate_angle_magnitude(hole_pos, QUAT_ANGLES_HOLE, MIN_BBOX_HOLE, MAX_BBOX_HOLE)
        #r_align = 1 - (angle_mag_peg + angle_mag_hole)

            # Orientation reward
        reward += 1 - np.tanh(d)
        reward += 1 - np.tanh(np.abs(t))
        reward += cos
        r_align = reward * 0.85

        r_insert = self._check_success()
        
        return  r_align, r_insert, r_reach #r_grasp, 

    def _get_env_info(self, action):
        #we have to do some changes for the two arm problem
        info = super()._get_env_info(action)
        r_align, r_insert, r_reach = self.staged_rewards() #r_grasp,   r_reach,
        t, d, cos = self._compute_orientation()
        info.update({
            'r_reach': r_reach / 0.10,
            #'r_grasp': r_grasp / 0.15,
            'r_align': r_align / 0.85,
            'success': r_insert,
            'align_t_abs': np.abs(t),
            'align_d': d,
            'align_cos': cos,
        })
        return info
    def _get_skill_info(self, robot_num):
        #we have to do some changes for the two arm problem
        #peg_tip_pos = self.sim.data.get_site_xpos(self.peg.important_sites["tip"])
        #hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
        #lift_pos = hole_pos + [0.0, self.task_config['lift_pos_offset'], 0.0]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
        dist = gripper_site_pos - hole_pos
        pos_info = {}
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])
        info = {}
        if robot_num == 0:
        #when peg goal pos is hole pos
            pos_info['push'] = [center] # push target positions
            pos_info['reach'] = [center] # reach target positions
            info['cur_ee_pos'] = peg_pos
        else:
        #when hole goal pos is peg pose
            pos_info['push'] = [peg_pos] # push target positions
            pos_info['reach'] = [peg_pos] # reach target positions
            info['cur_ee_pos'] = center
        
        for k in pos_info:
            info[k + '_pos'] = pos_info[k]

        return info

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
      
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # Add arena and robot
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0666432116509934, 1.4903257668114777e-08, 2.0563394967349096],
            quat=[0.6530979871749878, 0.27104058861732483, 0.27104055881500244, 0.6530978679656982],
        )

        # initialize objects of interest
        self.hole = PlateWithHoleObject(name="hole")
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.peg = CylinderObject(
            name="peg",
            size_min=(self.peg_radius[0], self.peg_length),
            size_max=(self.peg_radius[1], self.peg_length),
            material=greenwood,
            rgba=[0, 1, 0, 1],
            joints=None,
        )

        # Load hole object
        hole_obj = self.hole.get_obj()
        hole_obj.set("quat", "0 0 0.707 0.707")
        hole_obj.set("pos", "0.11 0 0.17")

        # Load peg object
        peg_obj = self.peg.get_obj()
        peg_obj.set("pos", array_to_string((0, 0, self.peg_length)))

        # Append appropriate objects to arms
        if self.env_configuration == "bimanual":
            r_eef, l_eef = [self.robots[0].robot_model.eef_name[arm] for arm in self.robots[0].arms]
            r_model, l_model = [self.robots[0].robot_model, self.robots[0].robot_model]
        else:
            r_eef, l_eef = [robot.robot_model.eef_name for robot in self.robots]
            r_model, l_model = [self.robots[0].robot_model, self.robots[1].robot_model]
        r_body = find_elements(root=r_model.worldbody, tags="body", attribs={"name": r_eef}, return_first=True)
        l_body = find_elements(root=l_model.worldbody, tags="body", attribs={"name": l_eef}, return_first=True)
        r_body.append(peg_obj)
        l_body.append(hole_obj)

        # task includes arena, robot, and objects of interest
        # We don't add peg and hole directly since they were already appended to the robots
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.hole)
        self.model.merge_assets(self.peg)
        print(self.model)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            if self.env_configuration == "bimanual":
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
            else:
                pf0 = self.robots[0].robot_model.naming_prefix
                pf1 = self.robots[1].robot_model.naming_prefix
            modality = "object"

            # position and rotation of peg and hole
            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hole_body_id])

            @sensor(modality=modality)
            def hole_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.hole_body_id], to="xyzw")

            @sensor(modality=modality)
            def peg_to_hole(obs_cache):
                return (
                    obs_cache["hole_pos"] - np.array(self.sim.data.body_xpos[self.peg_body_id])
                    if "hole_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")

            # Relative orientation parameters
            @sensor(modality=modality)
            def angle(obs_cache):
                t, d, cos = self._compute_orientation()
                obs_cache["t"] = t
                obs_cache["d"] = d
                return cos

            @sensor(modality=modality)
            def t(obs_cache):
                return obs_cache["t"] if "t" in obs_cache else 0.0

            @sensor(modality=modality)
            def d(obs_cache):
                return obs_cache["d"] if "d" in obs_cache else 0.0

            sensors = [hole_pos, hole_quat, peg_to_hole, peg_quat, angle, t, d]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        t, d, cos = self._compute_orientation()
 
        return  d < 0.15 and -0.22 <= t <= 0.24 and cos > 0.25 # d < 0.08 and -0.12 <= t <= 0.14 and cos > 0.95

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - peg_pos) @ v / (np.linalg.norm(v) ** 2)
        hole_normal = hole_mat @ np.array([0, 0, 1])
        d = np.linalg.norm(np.cross(hole_normal, peg_pos - center)) / np.linalg.norm(v) #v

        
        return (
            t,
            d,
            abs(np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)),
        )
    


    def _peg_pose_in_hole_frame(self):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.

        Returns:
            np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos(self.peg.root_body)
        peg_rot_in_world = self.sim.data.get_body_xmat(self.peg.root_body).reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        # World frame
        hole_pos_in_world = self.sim.data.get_body_xpos(self.hole.root_body)
        hole_rot_in_world = self.sim.data.get_body_xmat(self.hole.root_body).reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(peg_pose_in_world, world_pose_in_hole)
        return peg_pose_in_hole
