import numpy as np



def multiscale_maximum(img, start_point, scales=[4, 2, 1]):
    current_point = np.array(start_point, dtype=float)

    optimPath = []
    
    for scale in scales:
        optimPath.append(current_point.copy())
        # Downsample image
        scaled_img = img[::scale, ::scale]
        scaled_point = current_point / scale
        
        # Find optimum at this scale
        opt_point, _,  = hill_climb_discrete(scaled_img, scaled_point)
        
        # Scale back up
        current_point = opt_point * scale
    
    return current_point, img[int(current_point[0]), int(current_point[1])], optimPath

def hill_climb_discrete(img, start_point):
    current = [int(start_point[0]), int(start_point[1])]
    current_value = img[current[0], current[1]]
    
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while True:
        best_neighbor = current
        best_value = current_value
        
        for dy, dx in directions:
            ny, nx = current[0] + dy, current[1] + dx
            
            if (0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]):
                neighbor_value = img[ny, nx]
                if neighbor_value > best_value:
                    best_neighbor = [ny, nx]
                    best_value = neighbor_value
        
        if best_neighbor == current:  # No improvement found
            break
            
        current = best_neighbor
        current_value = best_value
    
    return np.array(current, dtype=float), current_value
