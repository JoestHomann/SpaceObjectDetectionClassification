```mermaid
graph LR
    %% --- STYLING ---
    classDef backbone fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:black;
    classDef head fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef act fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black;
    classDef tensor fill:#e0e0e0,stroke:#333,stroke-dasharray: 5 5,color:black;

    %% --- NODES ---

    Input[("Input Image")]:::tensor

    %% Collapsed Backbone Node
    Backbone["Backbone (ResNet18 Layers 0-4) Output: 512 channels"]:::backbone

    %% Heads Container
    subgraph "Custom Heads (1x1 Convs)"
        direction TB
        
        %% Center Branch
        subgraph Center_Branch [Center]
            C_Head["Center Head"]:::head
            C_Act["Sigmoid"]:::act
            C_Out[("Pred: (1, H, W)")]:::tensor
        end

        %% Box Branch
        subgraph Box_Branch [Box]
            B_Head["Box Head"]:::head
            B_Act["Sigmoid"]:::act
            B_Out[("Pred: (4, H, W)")]:::tensor
        end

        %% Class Branch
        subgraph Class_Branch [Class]
            K_Head["Class Head"]:::head
            K_Out[("Pred: (N, H, W)")]:::tensor
        end
    end

    %% --- WIRING ---
    Input --> Backbone
    Backbone -- "Feature Map (512, H, W)" --> C_Head & B_Head & K_Head
    
    C_Head --> C_Act --> C_Out
    B_Head --> B_Act --> B_Out
    K_Head --> K_Out