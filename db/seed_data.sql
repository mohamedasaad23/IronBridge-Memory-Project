INSERT INTO sites (id, name, near_power_lines) VALUES
  (1, 'Downtown Bridge Span A', 1),
  (2, 'Highway Interchange B', 0),
  (3, 'River Crossing C', 1);

INSERT INTO workers (id, name, role) VALUES
  (1, 'Ahmed Hassan', 'worker'),
  (2, 'Sara Nabil',   'worker'),
  (3, 'Omar Khaled',  'supervisor');

INSERT INTO equipment (id, site_id, name, type, high_risk, status) VALUES
  (1, 1, 'Tower Crane TC-40',   'CRANE',     1, 'AVAILABLE'),
  (2, 1, 'Excavator EX-220',    'EXCAVATOR', 0, 'AVAILABLE'),
  (3, 2, 'Mobile Crane MC-25',  'CRANE',     1, 'AVAILABLE'),
  (4, 2, 'Scaffold Set S-12',   'SCAFFOLD',  0, 'AVAILABLE'),
  (5, 3, 'Generator G-50',      'GENERATOR', 0, 'MAINTENANCE');

INSERT INTO certifications (worker_id, equipment_type, valid_until) VALUES
  (1, 'CRANE',     '2027-06-30'),
  (1, 'EXCAVATOR', '2027-03-15'),
  (2, 'CRANE',     '2025-01-10'),   -- expired
  (2, 'SCAFFOLD',  '2027-12-01'),
  (3, 'CRANE',     '2028-01-01');
