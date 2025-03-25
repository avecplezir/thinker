envs="MinAtar/Seaquest-v0 MinAtar/Asterix-v0 MinAtar/Breakout-v0 MinAtar/SpaceInvaders-v0 MinAtar/Freeway-v0"
agents="AgentActorConvETNoisy"

for agent in $agents
do
    for env in $envs
        do
            job_name="${agent}-${env}"
            mkdir -p out/
            run_cmd="scripts/run_atari.sh ${agent} ${env}"
            sbatch_cmd="sbatch --job-name $job_name ${run_cmd}"
            cmd="$sbatch_cmd"
            echo -e "${cmd}"
            ${cmd}
            sleep 1
        done
done

