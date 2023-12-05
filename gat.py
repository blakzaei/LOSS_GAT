#-- IMPORTS -------------------------------------------------------------------
import torch
from torch_geometric.nn import GATv2Conv
###############################################################################




#-- Class GAT Network ---------------------------------------------------------
class GAT_Network(torch.nn.Module):
    
    def __init__(self, num_features , num_hidden , num_outs , num_heads):                 
        super().__init__()
        
        #-- gat layers --------------------------------------------------------
        self.conv1 = GATv2Conv(num_features,
                               num_hidden,
                               heads=num_heads,
                               dropout=0.2)

        self.conv2 = GATv2Conv(num_hidden*num_heads,
                               num_hidden//2,
                               heads=num_heads//2,
                               dropout=0.2)

        self.conv3 = GATv2Conv((num_hidden//2)*(num_heads//2),
                               num_hidden//4,
                               heads=1,
                               dropout=0.2)

        #-- linear layers -----------------------------------------------------
        self.fc1 = torch.nn.Linear(num_hidden//4, num_hidden//8)
        self.fc2 = torch.nn.Linear(num_hidden//8, num_outs)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        embeddings = x
        
        x = self.fc1(x).relu()
        x = self.fc2(x)
        
        return x , embeddings
    #--------------------------------------------------------------------------
###############################################################################








