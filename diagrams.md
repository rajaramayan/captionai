# System Architecture & UML Diagrams

This document contains all the UML and architectural diagrams for the "Image Captioning Using AI Technique" project, generated using Mermaid.js syntax.

## 1. Flowchart: System Workflow

```mermaid
flowchart TD
    A[User Uploads Image] --> B[Input Validation]
    B --> C{Is Format Supported?}
    C -->|No| D[Return Error: Invalid Format]
    C -->|Yes| E[Save Image to /uploads]
    E --> F[Image Preprocessing Resize 299x299, Normalize]
    F --> G[InceptionV3 Feature Extraction]
    G --> H[2048-d Feature Vector]
    H --> I[LSTM Caption Generation]
    I --> J[Predict Next Word iteratively]
    J --> K{Is End Token?}
    K -->|No| J
    K -->|Yes| L[Format Caption String]
    L --> M[Save Prediction to Database]
    M --> N[Display Output to User]
```

## 2. Use Case Diagram

```mermaid
usecase
    actor User as "Registered User"
    actor Admin as "Administrator"
    
    package "Image Captioning System" {
        usecase UC1 as "Register / Login"
        usecase UC2 as "Upload Image"
        usecase UC3 as "View Generated Caption"
        usecase UC4 as "Listen to Caption (TTS)"
        usecase UC5 as "View Prediction History"
        usecase UC6 as "Export History to CSV"
        usecase UC7 as "View System Dashboard"
        usecase UC8 as "Manage Users & Logs"
    }
    
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    User --> UC7
    
    Admin --> UC1
    Admin --> UC7
    Admin --> UC8
```

## 3. Database ER Diagram

```mermaid
erDiagram
    USER ||--o{ PREDICTION : "makes"
    USER {
        int id PK
        string username
        string email
        string password_hash
        boolean is_admin
    }
    PREDICTION {
        int id PK
        string image_filename
        text caption
        float confidence
        float prediction_time
        datetime date_created
        int user_id FK
    }
```

## 4. Sequence Diagram: Image Upload & Prediction

```mermaid
sequenceDiagram
    actor U as User
    participant F as Frontend (UI)
    participant R as Routes (Flask API)
    participant M as ML Utils (CNN/LSTM)
    participant DB as SQLite Database
    
    U->>F: Selects & Uploads Image
    F->>R: POST /upload (multipart/form-data)
    R->>R: Validate Extension & Save File
    R->>M: generate_caption(image_path)
    M->>M: Preprocess Image
    M->>M: Extract Features (InceptionV3)
    M->>M: Generate Sequence (LSTM)
    M-->>R: Returns (caption, confidence)
    R->>DB: Add Prediction Record
    DB-->>R: Commit Success
    R-->>F: JSON {caption, confidence, time}
    F->>U: Display Image and Caption
```

## 5. Component Diagram

```mermaid
componentDiagram
    package "Frontend Layer" {
        [HTML/Jinja2 Templates]
        [Bootstrap 5 / CSS]
        [JavaScript / Fetch API]
    }
    
    package "Backend Layer (Flask)" {
        [Authentication Module]
        [Route Handlers]
        [Configuration Module]
    }
    
    package "Deep Learning Layer" {
        [CNN Extractor]
        [LSTM Generator]
        [Text Preprocessor]
    }
    
    package "Data Layer" {
        [SQLAlchemy ORM]
        [SQLite Database]
        [File System /uploads]
    }

    [JavaScript / Fetch API] --> [Route Handlers] : HTTP POST/GET
    [Route Handlers] --> [Authentication Module] : Verifies Session
    [Route Handlers] --> [CNN Extractor] : Passes Image
    [CNN Extractor] --> [LSTM Generator] : Passes Features
    [Route Handlers] --> [SQLAlchemy ORM] : Save History
    [SQLAlchemy ORM] --> [SQLite Database] : Read/Write
```

## 6. Activity Diagram: LSTM Generation Loop

```mermaid
stateDiagram-v2
    [*] --> Initialize_Sequence
    Initialize_Sequence --> Pass_To_Model : Input [Feature Vector, startseq]
    Pass_To_Model --> Predict_Probabilities
    Predict_Probabilities --> ArgMax : Get Highest Prob Index
    ArgMax --> Index_To_Word : Map to Vocabulary
    Index_To_Word --> Check_Token
    
    Check_Token --> Append_To_Sequence : If Word != endseq
    Append_To_Sequence --> Pass_To_Model : Next Iteration
    
    Check_Token --> Finalize_Caption : If Word == endseq OR length >= MAX
    Finalize_Caption --> [*]
```
