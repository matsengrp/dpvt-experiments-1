from dpvtex.dpvt_zoo import get_model_params_path, get_trained_model_path, train_model



def get_trained_model_ckpt(model, train_data, param_id, device, timestamp, output_dir):
    path = get_trained_model_path(
        model, train_data, param_id, device, timestamp, output_dir
    )
    return f"{path}.ckpt"


def generate_trained_model_ckpts(
    model_names, train_data_names, param_ids, device, timestamp, output_dir
):
    return [
        get_trained_model_ckpt(
            model=model_name,
            train_data=train_data_name,
            param_id=param_id,
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        )
        for model_name in model_names
        for train_data_name in train_data_names
        for param_id in param_ids
    ]



def get_model_params_json(model, train_data, param_id, device, timestamp, output_dir):
    """
    Get path to hyperparameter JSON file
    """
    path = get_model_params_path(
        model, train_data, param_id, device, timestamp, output_dir
    )
    return f"{path}.json"




def main():
    data_nicknames_path = "my_data_nicknames.json"
    model_name = "TraverseNN"
    train_data_name = "larch_alisim_alignment_50_seq_100_sites_500_algnmnts_train_spr"
    param_id = "Param0"
    device = "gpu"
    timestamp = "TODAY"
    output_dir = "_output"
    trained_model = generate_trained_model_ckpts(model_name, train_data_name, param_id, device, timestamp, output_dir)
    hyperparameter_path = get_model_params_json(
        model_name, train_data_name, param_id, device, timestamp, output_dir
    )
    train_model(
        model_name=model_name,
        data_name=train_data_name,
        train_checkpoint=trained_model,
        device=device,
        hyperparameter_path=hyperparameter_path,
        profiling=False,
        timestamp=timestamp,
        param_id=param_id,
        output_dir=output_dir,
        data_nicknames_path=data_nicknames_path,
    )


if __name__ == "__main__":
    main()