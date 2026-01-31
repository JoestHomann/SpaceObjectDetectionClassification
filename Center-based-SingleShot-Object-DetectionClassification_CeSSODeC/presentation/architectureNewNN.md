```mermaid
graph LR
    %% --- STYLING ---
    classDef backbone fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:black;
    classDef head fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef act fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black;
    classDef tensor fill:#e0e0e0,stroke:#333,stroke-dasharray: 5 5,color:black;
    classDef note fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:black;

    %% --- NODES ---
    Input[("Input Image\n(B,3,320,320)")]:::tensor

    %% Collapsed Backbone Node
    Backbone["Backbone (ResNet18)\nOutput feature map"]:::backbone
    Feat[("Feature Map\n(B,512,10,10)")]:::tensor

    %% Heads Container
    subgraph "Custom Heads (1x1 Convs)"
        direction TB
        
        %% Center Branch (UPDATED: logits, no sigmoid)
        subgraph Center_Branch [Center]
            C_Head["Center Head\n1x1 conv -> 1 ch"]:::head
            C_Out[("Center logits\n(B,1,10,10)")]:::tensor
        end

        %% Box Branch (same)
        subgraph Box_Branch [Box]
            B_Head["Box Head\n1x1 conv -> 4 ch"]:::head
            B_Act["Sigmoid"]:::act
            B_Out[("Box pred (norm xywh)\n(B,4,10,10)")]:::tensor
        end

        %% Class Branch (same)
        subgraph Class_Branch [Class]
            K_Head["Class Head\n1x1 conv -> N ch"]:::head
            K_Out[("Class logits\n(B,N,10,10)")]:::tensor
        end
    end

    %% Optional decode node (top-level architecture behavior)
    Decode["Decode (single-object)\n(i*,j*) = argmax(center logits)\nRead box+class at (i*,j*)"]:::note

    %% --- WIRING ---
    Input --> Backbone --> Feat
    Feat --> C_Head & B_Head & K_Head
    
    C_Head --> C_Out
    B_Head --> B_Act --> B_Out
    K_Head --> K_Out

    %% Behavioral coupling via decoding
    C_Out --> Decode
    B_Out --> Decode
    K_Out --> Decode
