import torch

"""
    Create the following tensors:
        1. 3D tensor of shape 20x30x40 with all values = 0
        2. 1D tensor containing the even numbers between 10 and 100
"""
three_d_tensor = torch.zeros((20, 30, 40))  # 3D tensor of shape 20x30x40 with all values = 0
#print("3D Tensor Shape:", three_d_tensor.shape)
even_numbers_tensor = torch.arange(10, 101, 2)  # 1D tensor containing even numbers between 10 and 100
#print("1D Tensor of Even Numbers:", even_numbers_tensor)
"""
    
    Calculate:
        1. Sum of all elements of x
        2. Sum of the columns of x  (result is a 6-element tensor)
        3. Sum of the rows of x   (result is a 4-element tensor)
"""
x = torch.rand(4, 6)
sum_all_elements = torch.sum(x)  # Sum of all elements of x
print("Sum of all elements of x:", sum_all_elements)
sum_columns = torch.sum(x, dim=0)  # Sum of the columns of x (result is a 6-element tensor)
print("Sum of columns of x:", sum_columns)
sum_rows = torch.sum(x, dim=1)  # Sum of the rows of x (result is a 4-element tensor)
print("Sum of rows of x:", sum_rows)
"""
    Calculate cosine similarity between 2 1D tensor:
    x = torch.tensor([0.1, 0.3, 2.3, 0.45])
    y = torch.tensor([0.13, 0.23, 2.33, 0.45])
"""
def cosine_similarity_1d(x, y):
    x_norm = torch.norm(x)
    y_norm = torch.norm(y)
    dot_product = torch.dot(x, y)
    similarity = dot_product / (x_norm * y_norm)
    return similarity
cosine_similarity_1d_result = cosine_similarity_1d(torch.tensor([0.1, 0.3, 2.3, 0.45]),
                                                   torch.tensor([0.13, 0.23, 2.33, 0.45]))
print("Cosine Similarity (1D):", cosine_similarity_1d_result)
"""
    Calculate cosine similarity between 2 2D tensor:
    x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
    y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
"""
def cosine_similarity_2d(x, y):
    x_norm = torch.norm(x, dim=1, keepdim=True)
    y_norm = torch.norm(y, dim=1, keepdim=True)
    dot_product = torch.mm(x, y.t())
    similarity = dot_product / (x_norm * y_norm.t())
    return similarity

cosine_similarity_2d_result = cosine_similarity_2d(
    torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                  [-2.2268, 1.9799, 1.5682, 0.5850],
                  [ 1.2289, 0.5043, -0.1625, 1.1403]]),
    
    torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                  [-0.6679, 0.0793, -2.5842, -1.5123],
                  [ 1.1110, -0.1212, 0.0324, 1.1277]]))
print("Cosine Similarity (2D):", cosine_similarity_2d_result)

"""
    x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
    Make x become 1D tensor
    Then, make that 1D tensor become 3x4 2D tensor 
"""
x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
x_1d = x.flatten()  # Make x become 1D tensor
print("1D Tensor:", x_1d)

"""
    x = torch.rand(3, 1080, 1920)
    y = torch.rand(3, 720, 1280)
    Do the following tasks:
        1. Make x become 1x3x1080x1920 4D tensor
        2. Make y become 1x3x720x1280 4D tensor
        3. Resize y to make it have the same size as x
        4. Join them to become 2x3x1080x1920 tensor
"""
x = torch.rand(3, 1080, 1920)
y = torch.rand(3, 720, 1280)
x_4d = x.unsqueeze(0)  # Make x become 1x3x1080x1920 4D tensor
y_4d = y.unsqueeze(0)  # Make y become 1x3x720x1280 4D tensor
y_resized = torch.nn.functional.interpolate(y_4d, size=(1080, 1920), mode='bilinear', align_corners=False)  # Resize y to match x
joined_tensor = torch.cat((x_4d, y_resized), dim=0)
print("Joined Tensor Shape:", joined_tensor.shape)  # Should be 2x3x1080x1920

