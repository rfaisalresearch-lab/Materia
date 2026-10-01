/**
 * Atom inspector.
 *
 * Each section separates what is measured, tabulated, computed or illustrative.
 * The educational shell diagram is labelled as such and is never presented as
 * a picture of electron motion; the quantum sections carry the configuration,
 * quantum numbers, orbital densities and the computed local density of states.
 * @module inspector
 */

import { api } from './api.js';
import { set, state } from './state.js';
import { angularPlot, pauliGrid, radialPlot, shellDiagram, slaterZeff } from './shell.js';
import { $, clear, el, num, originTag, tierTag } from './util.js';

const SECTION_HELP = {
  identity: 'Element and site data. Element properties are tabulated literature ' +
    'values; the position and role come from the structural model.',
  nucleus: 'Nuclide data from AME2020 and IUPAC isotopic compositions. A scanning ' +
    'probe does not resolve isotopes; an assignment here is a model input.',
  electrons: 'Ground-state configuration from the Aufbau/Madelung rule with the ' +
    'experimentally established exceptions, plus Pauli and Hund occupancy.',
  orbitals: 'Hydrogen-like orbitals evaluated with a Slater effective nuclear ' +
    'charge: the correct shapes and nodal structure for a one-electron ion, ' +
    'an approximation for a many-electron atom.',
  shell: 'An educational electron-shell diagram. Electrons do not follow these ' +
    'circular paths; the only physical content is the shell occupancy.',
  bonds: 'Bonds are perceived from interatomic distances unless a solver supplied ' +
    'bond orders. Lengths and angles are exact geometry of the current structure.',
  ldos: 'Site-projected density of states from the last electronic-structure run.',
  energy: 'Forces from the last solver run, and the electrostatic potential, field ' +
    'and Coulomb force at this site from the newest point-charge run that matches ' +
    'the current structure. A per-atom energy is not defined for a many-body ' +
    'potential and is therefore not reported.',
  probe: 'How this atom appears in the current scanning-probe image.',
  record: 'The provenance of every value shown in this inspector.',
  python: 'The Python expressions that reproduce this panel.',
};

export class Inspector {
  constructor(bodyId, hintId) {
    this.body = $(bodyId);
    this.hint = $(hintId);
    this.section = 'identity';
  }

  setSection(name) { this.section = name; this.render(); }

  async load(atomId) {
    try {
      const data = await api.inspectAtom(atomId);
      set({ atom: data });
      try {
        const ldos = await api.atomLdos(atomId);
        set({ atomLdos: ldos });
      } catch { set({ atomLdos: null }); }
      this.render();
    } catch (err) {
      clear(this.body);
      this.body.append(el('div.note.blocked', { text: err.message }));
    }
  }

  clearAtom() { set({ atom: null, atomLdos: null }); this.render(); }

  render() {
    const a = state.atom;
    clear(this.body);
    this.hint.textContent = a ? `#${a.identity.id} ${a.identity.element}` : '';
    if (!a) {
      this.body.append(el('p.empty', { text:
        'No atom selected. Choose the Select tool and click an atom in the atomic ' +
        'model, or click a feature in the probe image.' }));
      return;
    }
    this.body.append(el('div.note.info', { text: SECTION_HELP[this.section] || '' }));
    const fn = this[`_${this.section}`];
    if (fn) fn.call(this, a);
  }

  _grid(rows) {
    const table = el('table.pgrid');
    for (const [label, value, opts] of rows) {
      if (value === undefined) continue;
      const td = el('td', opts && opts.txt ? { class: 'txt' } : {});
      if (value instanceof Node) td.append(value);
      else td.textContent = value === null ? '-' : String(value);
      if (opts && opts.accent) td.classList.add('accent');
      if (opts && opts.blocked) td.classList.add('blocked');
      const tr = el('tr', {}, [el('th', { text: label, title: (opts && opts.title) || '' }), td]);
      table.append(tr);
    }
    return table;
  }

  _title(text, tag) {
    const d = el('div.section-title', {}, [el('span', { text })]);
    if (tag) d.append(tag);
    return d;
  }

  _identity(a) {
    const i = a.identity;
    this.body.append(this._title('Site', originTag('calculated')));
    this.body.append(this._grid([
      ['atom id', `#${i.id}`],
      ['element', `${i.element} - ${i.name}`],
      ['atomic number Z', i.atomic_number],
      ['mass', `${num(i.mass_u, 6)} u`],
      ['position', `(${num(i.position_A[0], 5)}, ${num(i.position_A[1], 5)}, ` +
        `${num(i.position_A[2], 5)}) Å`],
      ['velocity', i.velocity_A_fs.every((v) => v === 0) ? 'at rest'
        : `(${i.velocity_A_fs.map((v) => num(v, 4)).join(', ')}) Å/fs`],
      ['lattice site', i.label],
      ['structural role', i.role],
      ['fixed during relaxation', i.fixed ? 'yes' : 'no'],
    ]));
    this.body.append(this._title('Element reference', originTag('reference')));
    this.body.append(this._grid([
      ['category', i.category],
      ['group / period / block', `${i.group ?? 'f-block'} / ${i.period} / ${i.block}`],
      ['standard atomic weight', i.standard_atomic_weight === null ? null
        : `${num(i.standard_atomic_weight, 7)} u`],
      ['covalent radius', i.covalent_radius_A === null ? null : `${num(i.covalent_radius_A, 3)} Å`],
      ['van der Waals radius', i.vdw_radius_A === null ? null : `${num(i.vdw_radius_A, 3)} Å`],
      ['electronegativity (Pauling)', i.electronegativity_pauling],
      ['first ionisation energy', i.ionization_energy_eV === null ? null
        : `${num(i.ionization_energy_eV, 5)} eV`],
      ['electron affinity', i.electron_affinity_eV === null ? 'not evaluated'
        : `${num(i.electron_affinity_eV, 4)} eV`],
    ]));
  }

  _nucleus(a) {
    const n = a.nucleus;
    this.body.append(this._title('Nuclide', originTag('reference')));
    if (!n.isotope) {
      this.body.append(el('div.note.blocked', { text:
        `No isotope data is tabulated for ${a.identity.element} in this build. ` +
        'The nucleus panel reports only what is in the shipped nuclide table; it ' +
        'does not estimate.' }));
    } else {
      const iso = n.isotope;
      this.body.append(this._grid([
        ['nuclide', `${a.identity.element}-${iso.mass_number}`],
        ['protons (Z)', iso.protons],
        ['neutrons (N)', iso.neutrons],
        ['nuclide mass', `${num(iso.atomic_mass_u, 9)} u`],
        ['natural abundance', iso.natural_abundance === null ? 'not naturally occurring'
          : `${num(iso.natural_abundance * 100, 5)} %`],
        ['nuclear spin I', iso.spin === null ? 'not evaluated' : iso.spin],
        ['stability', iso.stable ? 'observationally stable'
          : `half life ${iso.half_life_s.toExponential(3)} s`],
        ['assignment', n.assigned ? 'explicitly assigned in this model'
          : 'most abundant natural nuclide'],
      ]));
    }
    this.body.append(el('div.note', { text: n.note }));
    if (n.available_isotopes.length) {
      this.body.append(this._title('Tabulated isotopes'));
      const table = el('table.grid');
      table.append(el('thead', {}, el('tr', {}, [
        el('th', { text: 'A' }), el('th', { text: 'mass / u' }),
        el('th', { text: 'abundance' }), el('th', { text: 'I' }),
        el('th', { text: 'stable' }), el('th', { text: '' })])));
      const tb = el('tbody');
      for (const iso of n.available_isotopes) {
        const btn = el('button.tool.sm', { text: 'assign',
          title: `Set this atom to ${a.identity.element}-${iso.mass_number}`,
          onclick: async () => {
            await api.edit('set_isotope', { ids: [a.identity.id], mass_number: iso.mass_number });
            await this.load(a.identity.id);
          } });
        tb.append(el('tr', {}, [
          el('td', { text: iso.mass_number }),
          el('td', { text: num(iso.mass_u, 8) }),
          el('td', { text: iso.abundance === null ? '-' : `${num(iso.abundance * 100, 4)} %` }),
          el('td', { text: iso.spin === null ? '-' : String(iso.spin) }),
          el('td', { text: iso.stable ? 'yes' : 'no' }),
          el('td', {}, btn)]));
      }
      table.append(tb);
      this.body.append(el('div.table-wrap', { style: { maxHeight: '190px' } }, table));
    }
  }

  _electrons(a) {
    const e = a.electrons;
    this.body.append(this._title('Configuration',
      originTag(e.origin)));
    this.body.append(this._grid([
      ['charge state', `${e.charge_e > 0 ? '+' : ''}${num(e.charge_e, 3)} e`],
      ['electron configuration', e.configuration],
      ['noble-gas notation', e.configuration_short],
      ['shell occupancy (K, L, M, …)', e.shell_occupancy.join(', ')],
      ['valence electrons', e.valence_electrons],
      ['unpaired electrons (Hund)', e.unpaired_electrons],
      ['spin multiplicity 2S+1', e.spin_multiplicity],
      ['magnetic moment', `${num(e.magnetic_moment_muB, 4)} µʙ`],
      ['partial charge (solver)', e.partial_charge_e === 0 ? 'not computed'
        : `${num(e.partial_charge_e, 4)} e`],
      ['confidence', `${Math.round(e.confidence * 100)} %`,
        { title: e.note, accent: true }],
    ]));
    this.body.append(el('div.note', { text: e.note }));

    this.body.append(this._title('Subshells and quantum numbers'));
    const table = el('table.grid');
    table.append(el('thead', {}, el('tr', {}, [
      el('th', { text: 'subshell' }), el('th', { text: 'n' }), el('th', { text: 'l' }),
      el('th', { text: 'mₗ' }), el('th', { text: 'electrons' }),
      el('th', { text: 'capacity' })])));
    const tb = el('tbody');
    for (const sh of e.subshells) {
      tb.append(el('tr', {}, [
        el('td', { text: sh.label }), el('td', { text: sh.n }), el('td', { text: sh.l }),
        el('td', { text: sh.m_l.join(', ') }), el('td', { text: sh.electrons }),
        el('td', { text: sh.capacity })]));
    }
    table.append(tb);
    this.body.append(el('div.table-wrap', {}, table));

    this.body.append(this._title('Pauli exclusion and Hund’s rules',
      originTag('illustrative')));
    this.body.append(pauliGrid(e.subshells, e.spin_orbitals));
    this.body.append(el('div.note', { text: e.caveat }));
    this.body.append(el('div.note', { html:
      '<b>Aufbau</b> fills the lowest available subshell first. <b>Pauli exclusion</b> ' +
      'allows at most one electron per (n, l, mₗ, mₛ). <b>Hund’s first ' +
      'rule</b> singly occupies every mₗ of a subshell with parallel spin before ' +
      'pairing. These rules fix the occupancy shown above; they do not fix which ' +
      'particular mₗ a given electron occupies.' }));
  }

  _orbitals(a) {
    const e = a.electrons;
    const Z = a.identity.atomic_number;
    this.body.append(this._title('Radial probability density', originTag('estimated')));
    this.body.append(radialPlot(e.subshells, Z));
    this.body.append(el('div.note', { html:
      'Hydrogen-like radial functions R<sub>nℓ</sub>(r) evaluated with a ' +
      'Slater effective nuclear charge Z*, plotted as the radial probability ' +
      '4πr²|R|² weighted by occupancy. The number of radial nodes ' +
      '(n − ℓ − 1) and the ordering of the shells are correct; the ' +
      'amplitudes are an approximation for a many-electron atom. Use a Tier-3 ' +
      'solver for a real electron density.' }));

    this.body.append(this._title('Angular shape |Yₗₘ|² (xz section)',
      originTag('illustrative')));
    const row = el('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '6px',
      padding: '6px' } });
    const seen = new Set();
    for (const sh of e.subshells) {
      for (const ml of sh.m_l) {
        const key = `${sh.l}:${Math.abs(ml)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const cell = el('div', { style: { textAlign: 'center' } });
        cell.append(angularPlot(sh.l, ml, 118));
        cell.append(el('div.readout', { text: `${sh.label} mₗ=${ml >= 0 ? '+' : ''}${ml}` }));
        row.append(cell);
      }
    }
    this.body.append(row);
    this.body.append(el('div.note', { text:
      'Real spherical-harmonic cross-sections. These are the angular factors of ' +
      'the one-electron solutions; they are exact for the hydrogen atom and are ' +
      'the conventional shapes used for any atom.' }));

    this.body.append(this._title('Effective nuclear charge (Slater)'));
    const rows = e.subshells.map((sh) => [sh.label,
      `Z* = ${num(slaterZeff(Z, e.subshells, sh.n, sh.l), 4)} (Z = ${Z})`]);
    this.body.append(this._grid(rows));
    this.body.append(el('div.note', { text:
      'Slater’s rules, J. C. Slater, Phys. Rev. 36 (1930) 57. An empirical ' +
      'screening estimate, not a self-consistent field.' }));
  }

  _shell(a) {
    const e = a.electrons;
    const n = a.nucleus.isotope;
    this.body.append(this._title('Educational electron-shell diagram',
      originTag('illustrative')));
    this.body.append(shellDiagram({
      symbol: a.identity.element,
      protons: a.identity.atomic_number,
      neutrons: n ? n.neutrons : null,
      occupancy: e.shell_occupancy,
      charge: Math.round(e.charge_e),
      valence: e.valence_electrons,
    }));
    this.body.append(el('div.note.warn', { html:
      '<b>This is a teaching diagram, not a physical picture.</b> Electrons do not ' +
      'travel on circular orbits, and they have no definite position between ' +
      'measurements. The only physical content of this drawing is the number of ' +
      'electrons in each principal shell. For the scientifically meaningful ' +
      'quantities see the <b>Electrons</b> and <b>Orbitals</b> sections, which give ' +
      'the configuration, quantum numbers and probability densities.' }));
    this.body.append(this._grid([
      ['protons', a.identity.atomic_number],
      ['neutrons', n ? n.neutrons : 'no isotope assigned'],
      ['electrons', a.identity.atomic_number - Math.round(e.charge_e)],
      ['shells', e.shell_occupancy.length],
      ['valence shell occupancy', e.shell_occupancy[e.shell_occupancy.length - 1]],
      ['net charge', `${e.charge_e > 0 ? '+' : ''}${num(e.charge_e, 3)} e`],
    ]));
  }

  _bonds(a) {
    const env = a.environment;
    this.body.append(this._title('Local environment', originTag('calculated')));
    this.body.append(this._grid([
      ['coordination number', env.coordination],
      ['nearest neighbour', env.nearest_neighbour_A === null ? 'none within the cutoff'
        : `${num(env.nearest_neighbour_A, 5)} Å`],
      ['bond perception', env.bond_origin, { txt: true }],
    ]));
    if (env.bonds.length) {
      const table = el('table.grid');
      table.append(el('thead', {}, el('tr', {}, [
        el('th', { text: 'partner' }), el('th', { text: 'element' }),
        el('th', { text: 'length / Å' }), el('th', { text: 'order' }),
        el('th', { text: 'origin' })])));
      const tb = el('tbody');
      for (const b of env.bonds) {
        tb.append(el('tr', { onclick: () => this.load(b.partner_id),
          title: 'Inspect this neighbour' }, [
          el('td', { text: `#${b.partner_id}` }),
          el('td', { text: b.partner_element }),
          el('td', { text: num(b.length_A, 5) }),
          el('td', { text: num(b.order, 3) }),
          el('td', { text: b.origin })]));
      }
      table.append(tb);
      this.body.append(el('div.table-wrap', { style: { maxHeight: '170px' } }, table));
    }
    if (env.angles.length) {
      this.body.append(this._title('Bond angles at this site'));
      const table = el('table.grid');
      table.append(el('thead', {}, el('tr', {}, [
        el('th', { text: 'a' }), el('th', { text: 'vertex' }), el('th', { text: 'c' }),
        el('th', { text: 'angle / °' })])));
      const tb = el('tbody');
      for (const g of env.angles) {
        tb.append(el('tr', {}, [
          el('td', { text: `#${g.a}` }), el('td', { text: `#${a.identity.id}` }),
          el('td', { text: `#${g.c}` }), el('td', { text: num(g.angle_deg, 4) })]));
      }
      table.append(tb);
      this.body.append(el('div.table-wrap', { style: { maxHeight: '170px' } }, table));
    }
  }

  _ldos(a) {
    const l = state.atomLdos;
    this.body.append(this._title('Local density of states',
      originTag(l && l.supported ? 'calculated' : 'unsupported')));
    if (!l || !l.supported) {
      this.body.append(el('div.note.blocked', { text: (l && l.reason) ||
        'No local density of states has been computed. Run Solve › Electronic ' +
        'structure first.' }));
      return;
    }
    set({ plot: 'ldos' });
    this.body.append(el('div.note', { text:
      'Plotted in the secondary pane. The LDOS is the site-projected density of ' +
      'states from the last electronic-structure run; it is what a scanning ' +
      'tunnelling spectrum probes at this site.' }));
    const e = state.electronic;
    if (e) {
      this.body.append(this._grid([
        ['solver', e.solver],
        ['Fermi level', `${num(e.fermi_level_eV, 5)} eV`],
        ['states computed', e.eigenvalues.length],
        ['HOMO-LUMO gap', e.hl_gap_eV === undefined ? 'not applicable'
          : `${num(e.hl_gap_eV, 5)} eV`],
        ['partial charge here', num(
          e.partial_charges[(e.ldos.atom_ids || []).indexOf(a.identity.id)], 4) + ' e'],
      ]));
      this.body.append(el('div.note.warn', { text: e.hl_gap_note
        ? e.hl_gap_note.note : '' }));
    }
  }

  _energy(a) {
    const en = a.energy;
    this.body.append(this._title('Forces',
      originTag(en.force_eV_A[0] === null ? 'unsupported' : 'calculated')));
    if (en.force_eV_A[0] === null) {
      this.body.append(el('div.note.blocked', { text:
        'No forces are available: no solver has run on this structure yet. ' +
        'Run Solve › Energy or Solve › Relax.' }));
    } else {
      this.body.append(this._grid([
        ['Fₓ', `${num(en.force_eV_A[0], 5)} eV/Å`],
        ['Fᵧ', `${num(en.force_eV_A[1], 5)} eV/Å`],
        ['Fᶻ', `${num(en.force_eV_A[2], 5)} eV/Å`],
        ['|F|', `${num(en.force_magnitude_eV_A, 5)} eV/Å`],
        ['|F| in nN', `${num(en.force_magnitude_eV_A * 1.602177, 5)} nN`],
      ]));
    }
    this.body.append(el('div.note', { text: en.note }));

    const es = a.electrostatics;
    this.body.append(this._title('Electrostatics at this site',
      originTag(es ? es.origin : 'unsupported')));
    if (!es) {
      this.body.append(el('div.note.blocked', { text:
        'No electrostatics run matches the current structure and charges. Assign point '
        + 'charges and press Compute electrostatics in the Solvers panel.' }));
      return;
    }
    const vec = (v, unit) => (v === null || v === undefined ? 'not stored'
      : `(${v.map((x) => num(x, 4)).join(', ')}) ${unit}`);
    const mag = (v) => (v ? Math.hypot(v[0], v[1], v[2]) : null);
    this.body.append(this._grid([
      ['point charge', es.point_charges === null ? 'not stored' : `${num(es.point_charges, 5)} e`],
      ['charge model', es.charge_model],
      ['site potential φ', es.site_potential === null ? 'not stored' : `${num(es.site_potential, 7)} V`],
      ['site field E', vec(es.site_field, 'V/Å')],
      ['|E|', es.site_field ? `${num(mag(es.site_field), 5)} V/Å` : 'not stored'],
      ['Coulomb force qE', vec(es.forces, 'eV/Å')],
      ['boundary', es.boundary_conditions],
      ['run', es.run_id],
    ]));
    this.body.append(el('div.note', { text:
      'Potential and field at the nucleus from every other point charge and all periodic '
      + 'images, excluding the charge of the atom itself. Point charges carry no electronic '
      + 'screening or polarisation.' }));
  }

  _probe(a) {
    const scan = state.scan;
    this.body.append(this._title('Appearance in the probe image'));
    if (!scan) {
      this.body.append(el('div.note.blocked', { text:
        'No scan has been acquired. Press Scan on the toolbar.' }));
      return;
    }
    const p = a.identity.position_A;
    const [x0, y0, x1, y1] = scan.extent_A;
    const inside = p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1;
    this.body.append(this._grid([
      ['technique', `${scan.technique} · ${scan.mode}`],
      ['channel', `${scan.channel} / ${scan.units[scan.channel] || ''}`],
      ['atom lateral position', `(${num(p[0], 4)}, ${num(p[1], 4)}) Å`],
      ['inside the scan window', inside ? 'yes' : 'no'],
      ['depth below the top atom', `${num(
        (state.render ? state.render.bounds[1][2] : p[2]) - p[2], 4)} Å`],
    ]));
    this.body.append(el('div.note.warn', { text:
      'A maximum in the image is a maximum of the measured signal, not a nucleus. ' +
      'Subsurface atoms contribute exponentially less: the probe sees the ' +
      'outermost one or two layers.' }));
    this.body.append(el('button.tool', { text: 'Move tip here', onclick: () => {
      set({ tip: { x: p[0], y: p[1] } });
    } }));
  }

  _record(a) {
    this.body.append(this._title('Where every value came from'));
    const rows = [
      ['element and isotope data', 'reference - IUPAC 2021 weights, AME2020 masses'],
      ['electron configuration', `${a.electrons.origin} - Madelung rule with ` +
        'experimentally established exceptions'],
      ['orbital plots', 'estimated - hydrogen-like with Slater screening'],
      ['shell diagram', 'illustrative - teaching diagram only'],
      ['position, bonds, angles', 'calculated - exact geometry of the current model'],
      ['bond perception', 'estimated - covalent-radius distance criterion'],
      ['forces', a.energy.force_eV_A[0] === null ? 'not computed'
        : 'calculated - last solver run'],
      ['local density of states', state.atomLdos && state.atomLdos.supported
        ? 'calculated - last electronic run' : 'not computed'],
    ];
    this.body.append(this._grid(rows.map(([k, v]) => [k, v, { txt: true }])));
    if (state.electronic && state.electronic.provenance) {
      const p = state.electronic.provenance;
      this.body.append(this._title('Last electronic solver', tierTag(p.fidelity)));
      this.body.append(this._grid([
        ['model', p.model],
        ['origin', p.origin],
        ['boundary conditions', p.boundary_conditions],
        ['software version', p.software_version],
      ]));
      const ul = el('ul', { style: { margin: '4px 10px', paddingLeft: '16px',
        fontSize: 'var(--fs-sm)', lineHeight: '1.5' } });
      for (const ap of p.approximations || []) ul.append(el('li', { text: ap }));
      this.body.append(ul);
      if ((p.references || []).length) {
        this.body.append(this._title('References'));
        const refs = el('ul', { style: { margin: '4px 10px', paddingLeft: '16px',
          fontSize: 'var(--fs-sm)', lineHeight: '1.5' } });
        for (const r of p.references) refs.append(el('li', { text: r }));
        this.body.append(refs);
      }
    }
  }

  _python(a) {
    const id = a.identity.id;
    const code = [
      '# Reproduce this panel from the Python console',
      'region = lab.active_region',
      `atom = region.atoms[${id}]`,
      'print(atom.symbol, atom.position, atom.charge)',
      'print(atom.electron_configuration)',
      'print("coordination:", atom.coordination)',
      'for n in atom.neighbors:',
      '    print(n.id, n.symbol, measure.distance(atom.id, n.id, region))',
      '',
      '# Electronic structure and the site-projected DOS',
      'out = region.solve()',
      'ldos = out["local_dos"].value',
      `k = list(ldos["atom_ids"]).index(${id})`,
      'view.plot(ldos["energy_eV"], ldos["ldos"][:, k],',
      `          title="LDOS of atom ${id}", xlabel="E / eV", ylabel="states/eV")`,
    ].join('\n');
    this.body.append(el('div.note', { text:
      'These expressions use the same API the interface uses. Copy them into the ' +
      'Python console below.' }));
    this.body.append(el('pre', { style: { margin: '6px 8px', padding: '8px',
      background: 'var(--panel-alt)', border: '1px solid var(--border)',
      fontFamily: 'var(--code)', fontSize: 'var(--fs-mono)', whiteSpace: 'pre-wrap',
      lineHeight: '1.5' }, text: code }));
    this.body.append(el('button.tool', { text: 'Send to console', onclick: () => {
      const editor = $('#script-editor');
      editor.value = code;
      document.querySelector('#bottom-dock .dock-tab[data-panel="console"]').click();
      editor.focus();
    } }));
  }
}
