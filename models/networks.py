import torch
from torch import nn
import torch.nn.functional as F

class MLP_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.fc1 = nn.Linear(512, 128).cuda()
        #self.fc2 = nn.Linear(128, 16).cuda()

        ####self.fc = nn.Linear(512, 32).cuda()
        self.fc = nn.Linear(512, 512).cuda()
        
        # Set the layer to be non-learnable
        # for param in self.fc.parameters():
        #     param.requires_grad = False
        
        seed = 42
        torch.manual_seed(seed)
        ####self.fixed_weights = torch.randn(32, 512).cuda()
        ####self.fixed_bias = torch.randn(32).cuda()
        self.fixed_weights = (torch.randn(512, 512)*0.001).cuda()
        self.fixed_bias = (torch.randn(512)*0.001).cuda()

        #print('max###', self.fixed_weights.max())
        #print('min###', self.fixed_weights.min())

        self.fc.weight = nn.Parameter(self.fixed_weights, requires_grad=False)
        self.fc.bias = nn.Parameter(self.fixed_bias, requires_grad=False)

    def forward(self, x):
        #x = torch.relu(self.fc1(x))    
        
        #x = self.fc1(x)
        #x = self.fc2(x)

        x = self.fc(x)

        return x
    

# class MLP_encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(512, 512).cuda()

#         # Set the weight and bias such that it performs an identity operation
#         with torch.no_grad():
#             self.fc.weight.fill_(0.01)  # Adjust the weight to control the scaling
#             self.fc.bias.fill_(0.0)  # No bias is added for simplicity

#     def forward(self, x):

#         #x = self.fc(x)

#         return x



def sf_linear_transformation(input_tensor, output_dim=512):
    seed = 42
    torch.manual_seed(seed)
    transformation_matrix = torch.randn(input_tensor.shape[0], input_tensor.shape[-1], output_dim).requires_grad_(False).cuda()
    output_tensor = torch.bmm(input_tensor, transformation_matrix)
    
    return output_tensor 

def fmap_linear_transformation(input_tensor, output_dim):
    seed = 42
    torch.manual_seed(seed)
    transformation_matrix = torch.randn(input_tensor.shape[0], output_dim).requires_grad_(False).cuda() #(16, 512)
    reshaped_tensor = input_tensor.view(input_tensor.shape[0], -1) #(16,360,480) -> (16, 360*480)
    transformed_tensor = torch.matmul(transformation_matrix.T, reshaped_tensor) # (512, 16) (16, 360*480) -> (512, 360*480)
    output_tensor = transformed_tensor.view(-1, input_tensor.shape[1], input_tensor.shape[2])
    
    return output_tensor 

########################################################################################

class MLP_decoder(nn.Module):
    def __init__(self, feature_out_dim):
        super().__init__()
        self.output_dim = feature_out_dim
        #self.fc1 = nn.Linear(16, 128).cuda()
        #self.fc1 = nn.Linear(128, 128).cuda()
        #self.fc2 = nn.Linear(128, self.output_dim).cuda()

        #self.fc0 = nn.Linear(16, 16).cuda()
        #self.fc1 = nn.Linear(16, 32).cuda()
        #self.fc2 = nn.Linear(32, 64).cuda()
        #self.fc3 = nn.Linear(128, 128).cuda()
        self.fc4 = nn.Linear(128, 256).cuda()

    def forward(self, x):
        input_dim, h, w = x.shape
        x = x.permute(1,2,0).contiguous().view(-1, input_dim) #(16,48,64)->(48,64,16)->(48*64,16)
        #x = torch.relu(self.fc0(x))
        #x = torch.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = torch.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(h, w, self.output_dim).permute(2, 0, 1).contiguous()
        return x


class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.input_dim = input_dim
        #self.output_dim = output_dim

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()


    def forward(self, x):
        
        x = self.conv(x)

        return x