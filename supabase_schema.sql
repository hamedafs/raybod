-- SQLAlchemy creates the table automatically on first startup.
-- This reference schema is provided for review only.
create table if not exists investor_assessments (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  investor_name text not null,
  risk_level text not null,
  risk_score integer not null,
  investor_profile_name text not null,
  report text not null,
  data_json text not null
);
