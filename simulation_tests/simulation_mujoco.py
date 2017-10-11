
import gym
import numpy as np
import time

NUM_TRIALS = 10
NUM_STEPS = 100

# LOAD ENVIRONMENT
if 'env' in locals(): 
    env.render(close=True)
env = gym.make('ReacherOneShot-v0')

# DEFINE PARAMETER RANGE
range1 = np.linspace(0, 2, 50)
range2 = np.linspace(0, -1, 50)

def executeTrial(t, param):
    theta_list = np.array([np.linspace(0, param[0], NUM_STEPS), np.linspace(0, param[1], NUM_STEPS)]).T
    init_pos = env.reset()
    init_pos = init_pos[:2]
    # EXECUTE TRIAL
    contact_cnt = 0
    for i in range(NUM_STEPS):
        env.render()
        control = init_pos + theta_list[i]
        ob, rew, done, info = env.step(control) 
        ball_xy = ob[2:4]
        # Check collision
        if env.unwrapped.data.ncon:
            contact_cnt+=1
            if contact_cnt > 5 and not sum(ball_xy)>0:
                return -1
    # Check movement
    if not sum(ball_xy)>0:
        return -2
    # Calculate polar coords
    ball_polar = np.array([ np.rad2deg(np.arctan2(-ball_xy[0], ball_xy[1])), np.sqrt(ball_xy[0]**2 + ball_xy[1]**2) * 100])
    # INFO
    print("\nTRIAL #", t+1,"params",[param1,param2])
    print("CTRL:\t",  control)
    print("Q_POS:\t", ob[:2])
    print("Q_VEL:\t", ob[4:6])
    print("BALL:\t",  ball_xy, ball_polar)
    print("Wsadjsada", ob[-1])
    
    return ball_polar



# RUN TRIALS
for t in range(3):
    param1 = np.random.choice(range1)
    param2 = np.random.choice(range2)

    res = executeTrial(t, [param1, param2])
    
    print(res)
    
