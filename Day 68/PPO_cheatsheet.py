# Init policy π_θ (actor) and value function V_φ (critic)

for iteration in range(N_ITER):

    # Rollout
    batch = []
    while len(batch) < BATCH_SIZE:
        state = env.reset()
        done = False
        while not done:
            action, log_prob = π_θ(state)                           # Sample action + log prob
            value = V_φ(state)                                      # Value estimate
            next_state, reward, done = env.step(action)
            batch.append((state, action, reward, done, log_prob, value))
            state = next_state

    # GAE advantage + returns
    advantages, returns = compute_gae(batch, γ, λ)

    # PPO update
    for epoch in range(N_EPOCHS):
        for minibatch in batch:
            new_log_prob = π_θ(minibatch.state).log_prob(minibatch.action)   # New log prob & entropy
            entropy = π_θ(minibatch.state).entropy()

            ratio = exp(new_log_prob - minibatch.old_log_prob)    # Clipped surrogate objective
            clip_obj = clip(ratio, 1 - ε, 1 + ε) * advantage
            policy_loss = -min(ratio * adv, clip_obj)

            value_loss = (V_φ(state) - return) ** 2                
            loss = policy_loss + c1 * value_loss - c2 * entropy   # Total loss: policy + value - entropy bonus

            update_parameters(loss)                               # Backprop + optimizer step


#key Hyper params

γ = 0.99         # Discount factor
λ = 0.95         # GAE lambda
ε = 0.2          # PPO clip epsilon
c1 = 0.5         # Value loss coefficient
c2 = 0.01        # Entropy coefficient
actor_lr = 3e-4
critic_lr = 1e-3
