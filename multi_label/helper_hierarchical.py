def map_predictions_to_quality(predictions, surface_type):
    quality_mapping = {
        "asphalt": [0, 1, 2, 3, 4, 5, 6, 7],  # Modify as needed
        "concrete": [4, 5, 6, 7, 8, 9],
        "paving_stones": [8, 9, 10, 11, 12, 13, 14, 15],
        "sett": [12, 13, 14, 15, 16],
        "unpaved": [15, 16, 17, 18,]
    }
    return torch.tensor([quality_mapping[surface_type][pred] for pred in predictions], dtype=torch.long)


def compute_fine_losses(fine_output, fine_labels_mapped, masks, head):
    fine_loss = 0.0
    
    if head == 'regression':
        fine_output_asphalt = fine_output[:, 0:1].float()
        fine_output_concrete = fine_output[:, 1:2].float()
        fine_output_paving_stones = fine_output[:, 2:3].float()
        fine_output_sett = fine_output[:, 3:4].float()
        fine_output_unpaved = fine_output[:, 4:5].float()
    
    elif head == 'corn':
        fine_output_asphalt = fine_output[:, 0:3]
        fine_output_concrete = fine_output[:, 3:6]
        fine_output_paving_stones = fine_output[:, 6:9]
        fine_output_sett = fine_output[:, 9:11]
        fine_output_unpaved = fine_output[:, 11:13]
                # Separate the fine outputs
    else:
        fine_output_asphalt = fine_output[:, 0:4]
        fine_output_concrete = fine_output[:, 4:8]
        fine_output_paving_stones = fine_output[:, 8:12]
        fine_output_sett = fine_output[:, 12:15]
        fine_output_unpaved = fine_output[:, 15:18]
        
    
    # Extract the masks
    asphalt_mask, concrete_mask, paving_stones_mask, sett_mask, unpaved_mask = masks
    
    # Get the labels for each surface type
    fine_labels_mapped_asphalt = fine_labels_mapped[asphalt_mask]
    fine_labels_mapped_concrete = fine_labels_mapped[concrete_mask]
    fine_labels_mapped_paving_stones = fine_labels_mapped[paving_stones_mask]
    fine_labels_mapped_sett = fine_labels_mapped[sett_mask]
    fine_labels_mapped_unpaved = fine_labels_mapped[unpaved_mask]

    three_mask_sett = (fine_labels_mapped_sett != 3)
    fine_labels_mapped_sett = fine_labels_mapped_sett[three_mask_sett]
    
    three_mask_unpaved = (fine_labels_mapped_unpaved != 3)
    fine_labels_mapped_unpaved = fine_labels_mapped_unpaved[three_mask_unpaved]
    
    # Compute the loss for each surface type
    if head == 'clm':
        fine_loss_asphalt = nn.NLLLoss()(torch.log(fine_output_asphalt[asphalt_mask] + 1e-9), fine_labels_mapped_asphalt)
        fine_loss_concrete = nn.NLLLoss()(torch.log(fine_output_concrete[concrete_mask] + 1e-9), fine_labels_mapped_concrete)
        fine_loss_paving_stones = nn.NLLLoss()(torch.log(fine_output_paving_stones[paving_stones_mask] + 1e-9), fine_labels_mapped_paving_stones)
        fine_loss_sett = nn.NLLLoss()(torch.log(fine_output_sett[sett_mask][three_mask_sett] + 1e-9), fine_labels_mapped_sett)
        fine_loss_unpaved = nn.NLLLoss()(torch.log(fine_output_unpaved[unpaved_mask][three_mask_unpaved] + 1e-9), fine_labels_mapped_unpaved)
    elif head == 'corn':
        fine_loss_asphalt = model.fine_criterion(fine_output_asphalt[asphalt_mask], fine_labels_mapped_asphalt, 4)
        fine_loss_concrete = model.fine_criterion(fine_output_concrete[concrete_mask], fine_labels_mapped_concrete, 4)
        fine_loss_paving_stones = model.fine_criterion(fine_output_paving_stones[paving_stones_mask], fine_labels_mapped_paving_stones, 4)
        fine_loss_sett = model.fine_criterion(fine_output_sett[sett_mask][three_mask_sett], fine_labels_mapped_sett, 3)
        fine_loss_unpaved = model.fine_criterion(fine_output_unpaved[unpaved_mask][three_mask_unpaved], fine_labels_mapped_unpaved, 3)
    elif head == 'regression':
        fine_loss_asphalt = nn.MSELoss()(fine_output_asphalt[asphalt_mask].flatten(), fine_labels_mapped_asphalt.float())
        fine_loss_concrete = nn.MSELoss()(fine_output_concrete[concrete_mask].flatten(), fine_labels_mapped_concrete.float())
        fine_loss_paving_stones = nn.MSELoss()(fine_output_paving_stones[paving_stones_mask].flatten(), fine_labels_mapped_paving_stones.float())
        fine_loss_sett = nn.MSELoss()(fine_output_sett[sett_mask][three_mask_sett].flatten(), fine_labels_mapped_sett.float())
        fine_loss_unpaved = nn.MSELoss()(fine_output_unpaved[unpaved_mask][three_mask_unpaved].flatten(), fine_labels_mapped_unpaved.float())
             
    fine_loss_asphalt = torch.nan_to_num(fine_loss_asphalt, nan=0.0)
    fine_loss_concrete = torch.nan_to_num(fine_loss_concrete, nan=0.0)
    fine_loss_paving_stones = torch.nan_to_num(fine_loss_paving_stones, nan=0.0)
    fine_loss_sett = torch.nan_to_num(fine_loss_sett, nan=0.0)
    fine_loss_unpaved = torch.nan_to_num(fine_loss_unpaved, nan=0.0)

    
    # Combine the losses
    fine_loss += fine_loss_asphalt
    fine_loss += fine_loss_concrete
    fine_loss += fine_loss_paving_stones
    fine_loss += fine_loss_sett
    fine_loss += fine_loss_unpaved
    
    return fine_loss

def compute_fine_metrics(coarse_probs, fine_output, fine_labels, masks, head):
    # Separate the fine outputs
    if head == 'regression':
        fine_output_asphalt = fine_output[:, 0:1].float()
        fine_output_concrete = fine_output[:, 1:2].float()
        fine_output_paving_stones = fine_output[:, 2:3].float()
        fine_output_sett = fine_output[:, 3:4].float()
        fine_output_unpaved = fine_output[:, 4:5].float()
    
    elif head == 'corn':
        fine_output_asphalt = fine_output[:, 0:3]
        fine_output_concrete = fine_output[:, 3:6]
        fine_output_paving_stones = fine_output[:, 6:9]
        fine_output_sett = fine_output[:, 9:11]
        fine_output_unpaved = fine_output[:, 11:13]
        
    else:
        fine_output_asphalt = fine_output[:, 0:4]
        fine_output_concrete = fine_output[:, 4:8]
        fine_output_paving_stones = fine_output[:, 8:12]
        fine_output_sett = fine_output[:, 12:15]
        fine_output_unpaved = fine_output[:, 15:18]
    
    # Extract the masks
    asphalt_mask, concrete_mask, paving_stones_mask, sett_mask, unpaved_mask = masks
    
 # Initialize prediction tensor
    predictions = torch.zeros_like(fine_labels)

    if asphalt_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            asphalt_preds = torch.argmax(fine_output_asphalt[asphalt_mask], dim=1)
        elif head == 'regression':
            asphalt_preds = fine_output_asphalt[asphalt_mask].round().long()
            print(asphalt_preds)
        elif head == 'corn':
            asphalt_preds = corn_label_from_logits(fine_output_asphalt[asphalt_mask]).long()
        predictions[asphalt_mask] = map_predictions_to_quality(asphalt_preds, "asphalt")

    if concrete_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            concrete_preds = torch.argmax(fine_output_concrete[concrete_mask], dim=1)
        elif head == 'regression':
            concrete_preds = fine_output_concrete[concrete_mask].round().long()
        elif head == 'corn':
            concrete_preds = corn_label_from_logits(fine_output_concrete[concrete_mask]).long()
        predictions[concrete_mask] = map_predictions_to_quality(concrete_preds, "concrete")

    if paving_stones_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            paving_stones_preds = torch.argmax(fine_output_paving_stones[paving_stones_mask], dim=1)
        elif head == 'regression':
            paving_stones_preds = fine_output_paving_stones[paving_stones_mask].round().long()
        elif head == 'corn':
            paving_stones_preds = corn_label_from_logits(fine_output_paving_stones[paving_stones_mask]).long()
        predictions[paving_stones_mask] = map_predictions_to_quality(paving_stones_preds, "paving_stones")

    if sett_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            sett_preds = torch.argmax(fine_output_sett[sett_mask], dim=1)
        elif head == 'regression':
            sett_preds = fine_output_sett[sett_mask].round().long()
        elif head == 'corn':
            sett_preds = corn_label_from_logits(fine_output_sett[sett_mask]).long()
        predictions[sett_mask] = map_predictions_to_quality(sett_preds, "sett")

    if unpaved_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            unpaved_preds = torch.argmax(fine_output_unpaved[unpaved_mask], dim=1)
        elif head == 'regression':
            unpaved_preds = fine_output_unpaved[unpaved_mask].round().long()
        elif head == 'corn':
            unpaved_preds = corn_label_from_logits(fine_output_unpaved[unpaved_mask]).long()
        predictions[unpaved_mask] = map_predictions_to_quality(unpaved_preds, "unpaved")


    # Calculate accuracy
    correct = (predictions == fine_labels).sum().item()
    
    correct_1_off = ((predictions == fine_labels) | 
               (predictions == fine_labels + 1) | 
               (predictions == fine_labels - 1)).sum().item()
    
    return correct, correct_1_off