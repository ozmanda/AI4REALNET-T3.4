# Importance Weighted Actor-Learner Architecture - [IMPALA](http://arxiv.org/abs/1802.01561)
This is an implementation of the IMPALA workflow for Flatland environments.

The key characteristic of IMAPALA is it's asynchronicity, meaning that the workers continuously generate trajectories with a local version of the policy, called the **behaviour policy**. The learner gathers trajectories until it has sufficient samples to update the **target policy**. The weights of this policy are broadcasted to the workers, who occaisonally update their local policy. Because trajectories from old policies are used to update the new target policy, a correction term is introduced, the **v-trace correction**. 
