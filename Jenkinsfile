pipeline {
    agent { docker { image 'python:3.8-buster' } }
    stages {
        stage('build') {
            steps {
                sh 'python --version'
            }
        }
        stage('git') {
            steps {
                echo 'Git'
                git(
                    url: "https://github.com/YsmanKhamidullin/ML-OPS",
                    branch: "main",
                    changelog: true
                )
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing dependencies'
                dir('./lab1/') {
                    sh 'pip install -r requirements.txt'
                }
            }
        }

        stage('ML-Pipeline') {
            steps {
                echo 'Calling pipeline.sh'
                dir('./lab1/') {
                    sh './pipeline.sh'
                }
            }
        }
    }
}
