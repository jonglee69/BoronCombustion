"""Blender 3.5 cut-away renders of ReaxFF nanoparticles (real MD coordinates).
Usage: blender -b -P render_blender.py -- results/figure11_snapshots results/figure11_renders 64 [key ...]
Tested with Blender 3.5.1 (Cycles, CPU)."""
import bpy, sys, os, math
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
CUT, OUT, SAMPLES, KEYS = argv[0], argv[1], int(argv[2]), argv[3:]
os.makedirs(OUT, exist_ok=True)
VIEW_DIR = Vector((1.0, -1.0, 0.85)).normalized()   # camera looks into the removed octant (x>0, y<0, z>0)
STYLE = {2: ('B', (0.25, 0.69, 0.29), 1.00), 3: ('O', (0.89, 0.17, 0.13), 0.82), 1: ('H', (0.95, 0.95, 0.95), 0.50)}


def srgb2lin(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def material(name, rgb):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*srgb2lin(rgb), 1)
    b.inputs['Roughness'].default_value = 0.28
    b.inputs['Specular'].default_value = 0.55
    b.inputs['Clearcoat'].default_value = 0.25
    return m


def atoms_object(name, pts, radius, mat):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(p) for p in pts], [], []); me.update()
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    ng = bpy.data.node_groups.new(name + '_gn', 'GeometryNodeTree')
    ng.inputs.new('NodeSocketGeometry', 'Geometry'); ng.outputs.new('NodeSocketGeometry', 'Geometry')
    n, L = ng.nodes, ng.links
    gi, go = n.new('NodeGroupInput'), n.new('NodeGroupOutput')
    m2p = n.new('GeometryNodeMeshToPoints')
    ico = n.new('GeometryNodeMeshIcoSphere'); ico.inputs['Radius'].default_value = radius; ico.inputs['Subdivisions'].default_value = 3
    sm = n.new('GeometryNodeSetShadeSmooth'); inst = n.new('GeometryNodeInstanceOnPoints')
    setm = n.new('GeometryNodeSetMaterial'); setm.inputs['Material'].default_value = mat
    L.new(gi.outputs['Geometry'], m2p.inputs['Mesh']); L.new(m2p.outputs['Points'], inst.inputs['Points'])
    L.new(ico.outputs['Mesh'], sm.inputs['Geometry']); L.new(sm.outputs['Geometry'], inst.inputs['Instance'])
    L.new(inst.outputs['Instances'], setm.inputs['Geometry']); L.new(setm.outputs['Geometry'], go.inputs['Geometry'])
    ob.modifiers.new('inst', 'NODES').node_group = ng


def scene(R):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
    sc.render.film_transparent = True; sc.render.resolution_x = sc.render.resolution_y = 1400
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Filmic'; sc.view_settings.look = 'Medium High Contrast'
    sc.render.threads_mode = 'FIXED'; sc.render.threads = 64
    w = bpy.data.worlds.new('w'); w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.45
    sc.world = w
    cam = bpy.data.cameras.new('cam'); cam.lens = 85; cam.sensor_width = 36
    dist = 2.35 * R / math.tan(math.atan(18 / 85)) / 2 * 1.0
    co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
    co.location = VIEW_DIR * dist
    co.rotation_euler = (-VIEW_DIR).to_track_quat('-Z', 'Y').to_euler()
    key = bpy.data.lights.new('key', 'AREA'); key.energy = 5.5e4 * (R / 50) ** 2; key.size = 1.2 * R
    ko = bpy.data.objects.new('key', key); sc.collection.objects.link(ko)
    ko.location = Vector((0.4 * R, -2.6 * R, 2.4 * R)); ko.rotation_euler = (-ko.location).to_track_quat('-Z', 'Y').to_euler()
    fill = bpy.data.lights.new('fill', 'AREA'); fill.energy = 1.2e4 * (R / 50) ** 2; fill.size = 2 * R
    fo = bpy.data.objects.new('fill', fill); sc.collection.objects.link(fo)
    fo.location = Vector((2.6 * R, 0.6 * R, 0.8 * R)); fo.rotation_euler = (-fo.location).to_track_quat('-Z', 'Y').to_euler()
    return sc


for f in sorted(os.listdir(CUT)):
    key = f[:-4]
    if not f.endswith('.npz') or (KEYS and key not in KEYS):
        continue
    d = np.load(os.path.join(CUT, f)); t, x = d['t'], d['x']
    keep = ~((x[:, 0] > 0) & (x[:, 1] < 0) & (x[:, 2] > 0)); t, x = t[keep], x[keep]
    R = float(np.percentile(np.linalg.norm(x, axis=1), 99.5))
    sc = scene(R)
    for typ, (nm, rgb, rad) in STYLE.items():
        m = t == typ
        if m.any():
            atoms_object(nm, x[m], rad, material('m' + nm, rgb))
    sc.render.filepath = os.path.join(OUT, key + '.png')
    bpy.ops.render.render(write_still=True)
    print('RENDERED', key, len(t), 'R=%.1f' % R)
