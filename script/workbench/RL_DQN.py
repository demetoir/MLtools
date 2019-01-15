#
def e_greedy(e=0.1, decay=0.9):
    # return action
    e= e * decay
    if random < e:
        action = random_action
    else:
        action = argmax(q(s, a))

    return a

def add_noise():
    # nn 을 사용할거라면 벡터에 노이즈를 넣을수도있음
    # e_greedy 보다는 덜 랜덤함
    # 기존 q 값을 기반으로 함
    pass


def q_learning_algorithm():
    # q table learning 은 생략한다

    def policy(state):
        # return action by policy
        return 0

    def update_q_function(old_state, new_state, reward, action, lr, discount_factor):
        q = {}
        q[(old_state, action)] += lr * (
                reward + discount_factor * argmax(q[(new_state, new_action)]) - q[(old_state, action)])

        return

    env = None

    lr = 0.8
    discount_factor = 0.3
    nam_episode = 1000
    for i in range(nam_episode):
        env.reset()

        # get state from env
        state = env.get_init_state()

        while env.is_end():
            # get action from s by policy function
            action = policy(state)

            # do action and get reward and new state
            reward, new_state = env.do_action(action)

            # update q function
            # q(s,a) = r + q(s', a')
            update_q_function(state, new_state, reward, action, lr, discount_factor)

            # assign new state
            state = new_state

            pass
        pass
