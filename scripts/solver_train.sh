solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1



# echo 'start generate question'
# bash question_generate/question_generate.bash $questioner_model_path 70 $experiment_name 
# echo 'start evaluate generated question'
# bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
# echo 'start save dataset to local'
# python question_evaluate/save_local.py --repo_name ${experiment_name} --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name}
echo 'start train'

# python3 -m verl.trainer.main \
#     config=examples/config.yaml \
#     data.max_response_length=16384 \
#     worker.rollout.max_num_batched_tokens=24576 \
#     worker.actor.model.model_path=$solver_model_path \
#     trainer.experiment_name=${experiment_name} \
#     trainer.load_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/global_step_5 \
#     trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
#     data.train_files=${STORAGE_PATH}/generated_question/${experiment_name}_train/merge_data.json \
#     trainer.total_epochs=100 \
#     trainer.max_steps=6 \
#     data.format_prompt=./examples/format_prompt/kernelbench.jinja \
#     trainer.val_freq=-1 \
#     trainer.val_before_train=false \
#     worker.actor.micro_batch_size_per_device_for_update=1 \
#     worker.actor.micro_batch_size_per_device_for_experience=1 \
#     trainer.save_freq=3

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_6/actor

sleep 10

echo "solver training finished"

# bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
