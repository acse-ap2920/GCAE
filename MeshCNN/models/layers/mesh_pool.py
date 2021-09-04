import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify

def calculateangle(origin, end):
  x1, y1 = origin
  x2, y2 = end
  result = np.arctan2(y2-y1,x2-x1)
  if result<0:
    return result+2*np.pi
  else:
    return result

def checkcollapse(origin, new, points):
  oldorder = np.argsort([(calculateangle(origin,i)) for i in points])
  neworder = np.argsort([(calculateangle(new,i)) for i in points])

  duplicate = np.tile(oldorder,2)
  index = np.where(oldorder == neworder[0])[0][0]
  subpart = duplicate[index:index+len(oldorder)]
  if (neworder == subpart).all():
    return True
  else:
    return False

def findcyl(mesh):
  nocv = []
  for i in range(len(mesh.v_mask)):
    if -1 in mesh.gemm_edges[mesh.ve[i]]:
      vert = mesh.vs[i][:2]
      if vert[0]!= 0 and vert[1]!= 0 and vert[0]!= 2.2 and vert[1]!= 0.41:
        nocv.append(vert)
  return np.array(nocv)

def isnotedge(mesh,v,cylinder):
  notleft  = mesh.vs[v][0] != 0
  notbottom  = mesh.vs[v][1] != 0
  notright  = mesh.vs[v][0] != 2.2
  nottop  = mesh.vs[v][1] != 0.41
  notcyl = mesh.vs[v][:2] not in cylinder
  return notleft and notbottom and notright and nottop and notcyl

def collapseornot(mesh, edge, cylinder):
  v1, v2 = mesh.edges[edge]
  new = (mesh.vs[v1]+mesh.vs[v2])[:2]/2

  todelv1 = np.array([mesh.edges[edg] for edg in mesh.ve[v1]]).flatten()
  todelv2 = np.array([mesh.edges[edg] for edg in mesh.ve[v2]]).flatten()

  finalv1 = np.delete(todelv1, np.where(todelv1 == v1))
  finalv2 = np.delete(todelv2, np.where(todelv2 == v2))

  coordsv1 = [mesh.vs[v][:2] for v in finalv1]
  coordsv2 = [mesh.vs[v][:2] for v in finalv2]

  firstcondition = (checkcollapse(mesh.vs[v1][:2],new,coordsv1)) and (checkcollapse(mesh.vs[v2][:2],new,coordsv2))
  
  secondcondition = isnotedge(mesh,v1,cylinder) and isnotedge(mesh,v2,cylinder)
  
  return firstcondition and secondcondition

class MeshPool(nn.Module):
    
    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.edges_count], mesh.edges_count)
        # recycle = []
        # last_queue_len = len(queue)
        last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        cyl = findcyl(mesh)
        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            # if mask[edge_id] and self.has_boundaries(mesh,edge_id) and mesh.edges_count < 800:
              # print("collapsing boundary edge", edge_id)
              # self.__pool_boundary(mesh, edge_id, mask, edge_groups)
              # print("collapsed boundary edge", edge_id)
            if mask[edge_id] and collapseornot(mesh, edge_id, cyl):
                # print(mesh.gemm_edges[edge_id])
                # print("collapsing edge", edge_id)
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
                # print("collapsed edge", edge_id)
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0)\
            and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) \
            and self.__is_one_ring_valid(mesh, edge_id):
            # self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            # self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            mesh.merge_vertices(edge_id)
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False

    def __pool_boundary(self, mesh, edge_id, mask, edge_groups):
        try:
          self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
        except:
          self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
        mesh.merge_vertices(edge_id)
        mask[edge_id] = False
        MeshPool.__remove_group(mesh, edge_groups, edge_id)
        mesh.edges_count -= 1
        return True
  
    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False


    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1], mesh.sides[key_b, other_side_b + 1])
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert (len(shared_items) == 2)
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            MeshPool.__redirect_edges(mesh, update_key_a, MeshPool.__get_other_side(update_side_a), update_key_b, MeshPool.__get_other_side(update_side_b))
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert(len(vertex) == 1)
        mesh.remove_vertex(vertex[0])

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)

