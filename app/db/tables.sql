-- Never forget to add ONCASCADE to foreign keys that are not in the same table
-- and are not in the same table as the one that is being deleted
-- Too much trauma

-- USERS TABLE
CREATE TABLE IF NOT EXISTS Users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    password VARCHAR(1000) NOT NULL,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    secret_key INT CHECK (secret_key BETWEEN 1000 AND 9999) NOT NULL
);

-- PROJECTS TABLE (deletes when user is deleted)
CREATE TABLE IF NOT EXISTS Projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INT NOT NULL,
    CONSTRAINT fk_project_user
        FOREIGN KEY (user_id)
        REFERENCES Users(id)
        ON DELETE CASCADE
);

-- FILES TABLE (deletes when user is deleted)
CREATE TABLE IF NOT EXISTS Files (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(255) NOT NULL,
    user_id INT NOT NULL,
    CONSTRAINT fk_file_user
        FOREIGN KEY (user_id)
        REFERENCES Users(id)
        ON DELETE CASCADE
);

-- DATASETS TABLE (deletes when project, user, or file is deleted)
CREATE TABLE IF NOT EXISTS Datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    project_id INT NOT NULL,
    user_id INT NOT NULL,
    file_id INT NOT NULL,
    CONSTRAINT fk_dataset_project
        FOREIGN KEY (project_id)
        REFERENCES Projects(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_dataset_user
        FOREIGN KEY (user_id)
        REFERENCES Users(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_dataset_file
        FOREIGN KEY (file_id)
        REFERENCES Files(id)
        ON DELETE CASCADE
);

-- ML_Results TaBLE (deletes when project, user, or dataset is deleted)
CREATE TABLE IF NOT EXISTS ML_Results (
    id SERIAL PRIMARY KEY,
    project_id INT NOT NULL,
    user_id INT NOT NULL,
    job_type TEXT CHECK (job_type IN ('classification', 'regression', 'clustering')),
    model_type TEXT NOT NULL,
    target_ TEXT,
    features TEXT[], -- only used for clustering
    metrics JSONB, -- JSONB for storing metrics like accuracy, precision, recall, etc.

    plot_1 TEXT,  -- base64 string
    plot_2 TEXT,
    plot_3 TEXT,
    plot_4 TEXT,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_ml_result_project
        FOREIGN KEY (project_id)
        REFERENCES Projects(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_ml_result_user
        FOREIGN KEY (user_id)
        REFERENCES Users(id)
        ON DELETE CASCADE
);

