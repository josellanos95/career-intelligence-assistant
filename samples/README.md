# Sample data

Fictional resume and job descriptions for manually exercising the app and for the eval harness in
`evals/`. None of this describes a real person or a real company.

- `resumes/jordan_avery_resume.txt` -- a backend/AI engineer profile with a deliberate mix of strong
  matches and real gaps (no Kubernetes, no agent-orchestration framework experience, no MLOps).
- `job_descriptions/ai_platform_engineer_healthforward.txt` -- close fit: same domain (RAG, Python),
  gaps on Kubernetes and agent frameworks.
- `job_descriptions/senior_backend_engineer_ledgerpeak.txt` -- partial fit: strong on Python/backend,
  gaps on Kafka/event-driven systems and a secondary JVM language.
- `job_descriptions/ml_infrastructure_engineer_vertexrobotics.txt` -- poor fit: a genuinely different
  job (ML infra / computer vision), used to check the assistant doesn't force a positive spin on a bad
  match.

Upload the resume once, then upload each job description separately (`doc_type=job_description`) to try
skill-gap and interview-prep questions against a specific job via the "Focus on job" selector in the UI.
