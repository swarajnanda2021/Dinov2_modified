"""
Parameter groups handling for FSDP2 models.
"""

def remove_fsdp_compile_names(name: str) -> str:
    """
    Remove FSDP2, compile, and checkpointing prefixes from parameter names.
    """
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("_checkpoint_wrapped_module.", "")
    name = name.replace("parametrizations.", "")
    name = name.removesuffix(".original")
    name = name.replace("module.", "")
    name = name.replace("_orig_mod.", "")
    return name


def get_params_groups_fsdp2(model):
    """
    Get parameter groups with FSDP2-aware name handling.
    
    Args:
        model: FSDP2-wrapped model
        
    Returns:
        List of parameter group dictionaries
    """
    regularized = []
    not_regularized = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Clean FSDP2 prefixes
        clean_name = remove_fsdp_compile_names(name)
        
        # No weight decay on biases or 1D parameters (norms)
        if clean_name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    
    return [
        {'params': regularized},
        {'params': not_regularized, 'weight_decay': 0.}
    ]