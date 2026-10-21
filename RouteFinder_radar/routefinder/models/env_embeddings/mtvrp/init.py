import torch
import torch.nn as nn
import torch.nn.functional as F


class MTVRPInitEmbedding(nn.Module):
    """Initial embedding MTVRP.

    Note: this is the same as what MTPOMO and MVMoE use.

    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - durations: duration of the nodes (TW)

    Global features:
        - loc: x, y euclidean coordinates of depot
    """

    def __init__(
        self, embed_dim=128, bias=False, num_global_feats=2, num_cust_feats=7, **kw
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(MTVRPInitEmbedding, self).__init__()

        # Depot feats (includes global features): x, y, distance, backhaul_class, open_route
        # Customer feats: x, y, demand_linehaul, demand_backhaul, time_window_early, time_window_late, durations
        customer_feat_dim = num_cust_feats
        # self.project_customers_feats = nn.Linear(customer_feat_dim - 2, embed_dim, bias=bias)
        self.k = 10
        self.projection = nn.Linear(2 * self.k + customer_feat_dim - 2, embed_dim) 
        
        self.icam_projection = nn.Linear(50 + customer_feat_dim - 2, embed_dim) 
        self.icam_global_projection = nn.Linear(50, embed_dim) 
        
        self.global_projection = nn.Linear(2 * self.k, embed_dim) 

        # self.nan_global =  nn.Linear(num_global_feats, embed_dim) 
        self.nan_local = nn.Linear(customer_feat_dim, embed_dim)
          
        self.embed_dim = embed_dim

    def pre_forward(self, td):
        problems = td["locs"]
        mean_val = problems.mean(dim=(1, 2), keepdim=True)
        std_val = problems.std(dim=(1, 2), keepdim=True)
        problems = (problems - mean_val) / (std_val + 1e-9)  # 避免除0
        with torch.cuda.amp.autocast(enabled=False):
            problems = problems.to(dtype=torch.float32)
            
            U, S, V = torch.svd_lowrank(problems, q=self.k)
            sqrt_S = torch.sqrt(S)  # (batch, k)

        Q = U * sqrt_S.unsqueeze(1)  # (batch, n, k)
        K = V * sqrt_S.unsqueeze(1)  # (batch, n, k)

        self.X = torch.cat([Q, K], dim=-1)  # (batch, n, 2*k)
        

    def forward(self, td):
        # Global (batch, 1, 2) -> (batch, 1, embed_dim)
        # Customers (batch, N, 5) -> (batch, N, embed_dim)

        # note that these feats include the depot (but unused) so we exclude the first node
        # self.X = torch.randint(1, 10, self.X.shape, device=self.X.device).float() 
        # self.X = torch.randn_like(self.X).float() 

        # problems = td["locs"]
        # mean_val = problems.mean(dim=(1, 2), keepdim=True)
        # std_val = problems.std(dim=(1, 2), keepdim=True)
        # problems = (problems - mean_val) / (std_val + 1e-9)  # 避免除0
        # _, indices = torch.topk(problems, 50 + 1, largest=False, dim=-1)
        # indices = indices[:, :, 1:]  # 结果形状为 (batch, node, k)
        # self.X = torch.gather(problems, 2, indices)  # (batch, node, k)


        cust_feats = torch.cat(
            # TODO replace 1 with actual number of depots
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None]
            ),
            -1,
        )

        # If some features are infinity (e.g. distance limit is inf because of no limit), replace with 0 so that it does not affect the embedding
        # global_feats = torch.nan_to_num(self.final_embedding, nan=0.0, posinf=0.0, neginf=0.0)
        cust_feats = torch.nan_to_num(cust_feats, nan=0.0, posinf=0.0, neginf=0.0)
        cust_feats = torch.concat([self.X[:, 1:, :], cust_feats], -1)

        # cust_embeddings = self.nan_local(cust_feats)
        # cust_embeddings = self.icam_projection(cust_feats)
        cust_embeddings = self.projection(cust_feats)

        
        # # 1) 生成一个 (batch, 1, 2) 的随机张量（标准正态分布）
        global_embeddings = self.global_projection(self.X[:, 0:1, :])
        # global_embeddings = self.icam_global_projection(self.X[:, 0:1, :])

        # cust_embeddings = self.projection(
        #     cust_feats
        # )  # [batch, N, embed_dim]
        # cust_embeddings = cust_embeddings + self.final_embedding[:, 0:, :]

        return torch.cat(
            (global_embeddings, cust_embeddings), -2
        )  # [batch, N+1, embed_dim]


# Note that this is the most recent version and should be used from now on. Others can be based on this!
class MTVRPInitEmbeddingRouteFinderBase(nn.Module):
    """General Init embedding class

    Args:
        num_global_feats: number of global features
        num_cust_feats: number of customer features
        embed_dim: embedding dimension
        bias: whether to use bias in the linear layers
        posinf_val: value to replace positive infinity values with
    """
    # 将全局特征 (global_feats) 和客户特征 (cust_feats) 投影到相同的高维嵌入空间 (embed_dim)

    def __init__(
        self, num_global_feats, num_cust_feats, embed_dim=128, bias=False, posinf_val=0.0,
        init_type="svd",
    ):
        super(MTVRPInitEmbeddingRouteFinderBase, self).__init__()
        # distance features: "svd" (RADAR) / "nn" (50 nearest distances) / "none"
        assert init_type in ("svd", "nn", "none"), init_type
        self.init_type = init_type
        # num_global_feats -= 2
        # num_cust_feats -= 2
        # self.project_global_feats = nn.Linear(num_global_feats, embed_dim, bias=bias)
        # self.project_customers_feats = nn.Linear(num_cust_feats, embed_dim, bias=bias)
        self.embed_dim = embed_dim
        # 处理正无穷 (inf) 时的替换值 (默认为 0.0)
        self.posinf_val = posinf_val

        self.k = 10
        self.global_projection = nn.Linear(2 * self.k + num_global_feats, embed_dim) 
        self.local_projection = nn.Linear(2 * self.k + num_cust_feats, embed_dim)

        self.nan_global =  nn.Linear(num_global_feats, embed_dim) 
        self.nan_local = nn.Linear(num_cust_feats, embed_dim)

        self.icam_projection = nn.Linear(50 + num_cust_feats, embed_dim) 
        self.icam_global_projection = nn.Linear(50 + num_global_feats, embed_dim) 

    def _global_feats(self, td):
        # 留给子类去定义具体的特征提取逻辑
        raise NotImplementedError("This method should be overridden by subclasses")

    def _cust_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def pre_forward(self, td):
        problems = td["locs"]
        mean_val = problems.mean(dim=(1, 2), keepdim=True)
        std_val = problems.std(dim=(1, 2), keepdim=True)
        problems = (problems - mean_val) / (std_val + 1e-9)  # 避免除0
        
        with torch.cuda.amp.autocast(enabled=False):
            problems = problems.to(dtype=torch.float32)
            
            U, S, V = torch.svd_lowrank(problems, q=self.k)
            sqrt_S = torch.sqrt(S)  # (batch, k)

        Q = U * sqrt_S.unsqueeze(1)  # (batch, n, k)
        K = V * sqrt_S.unsqueeze(1)  # (batch, n, k)

        self.X = torch.cat([Q, K], dim=-1)  # (batch, n, 2*k)


    def forward(self, td):
        global_feats = self._global_feats(td)
        cust_feats = self._cust_feats(td)

        # replace +inf (e.g. no distance limit) with posinf_val
        global_feats = torch.nan_to_num(global_feats, posinf=self.posinf_val)
        cust_feats = torch.nan_to_num(cust_feats, posinf=self.posinf_val)

        init_type = getattr(self, "init_type", "svd")

        if init_type == "none":  # node features only
            global_embeddings = self.nan_global(global_feats)
            cust_embeddings = self.nan_local(cust_feats)
            return torch.cat((global_embeddings, cust_embeddings), -2)

        if init_type == "nn":  # standardized distances to the 50 nearest nodes (ICAM-style)
            problems = td["locs"]
            problems = (problems - problems.mean(dim=(1, 2), keepdim=True)) / (
                problems.std(dim=(1, 2), keepdim=True) + 1e-9
            )
            _, indices = torch.topk(problems, 50 + 1, largest=False, dim=-1)
            X = torch.gather(problems, 2, indices[:, :, 1:])  # (batch, node, 50)
            global_embeddings = self.icam_global_projection(torch.concat((X[:, 0:1, :], global_feats), -1))
            cust_embeddings = self.icam_projection(torch.concat((X[:, 1:, :], cust_feats), -1))
            return torch.cat((global_embeddings, cust_embeddings), -2)

        # RADAR: SVD-based distance features
        self.pre_forward(td)
        global_feats = torch.concat((self.X[:, 0:1, :], global_feats), -1)
        global_embeddings = self.global_projection(global_feats)
        cust_feats = torch.concat((self.X[:, 1:, :], cust_feats), -1)
        cust_embeddings = self.local_projection(cust_feats)
        return torch.cat((global_embeddings, cust_embeddings), -2)


class MTVRPInitEmbeddingRouteFinder(MTVRPInitEmbeddingRouteFinderBase):
    """
    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - service_time: service time of the nodes
    Global features:
        - open_route (O)
        - distance_limit (L)
        - (end) time window of depot
        - x, y euclidean coordinates of depot
    The above features are embedded in the depot node as global and get broadcasted via attention.
    This allows the network to learn the relationships between them.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0, init_type="svd"):
        super(MTVRPInitEmbeddingRouteFinder, self).__init__(
            num_global_feats=3,  # x, y, open_route, distance_limit, time_window_depot
            num_cust_feats=5,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
            init_type=init_type,
        )

    def _global_feats(self, td):
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )

    def _cust_feats(self, td):
        # [open_route, depot_x, depot_y, distance_limit, depot_time_window_end]
        return torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
            ),
            -1,
        )


class MTVRPInitEmbeddingM(MTVRPInitEmbeddingRouteFinder):
    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        # 增加了 backhaul_class 这一特征，以便模型能考虑**回程负载（backhaul）**的信息
        # Note: here we add the backhaul_class as a feature
        MTVRPInitEmbeddingRouteFinderBase.__init__(
            self,
            num_global_feats=3 + 1,
            num_cust_feats=5,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        # 调用 super()._global_feats(td) 获取 原有的 5 个全局特征
        # 纯送货 (backhaul_class=0)：车辆空载出发，满载返回。
        # 纯回程 (backhaul_class=1)：车辆空载出发，逐渐装载货物。
        # 混合任务 (backhaul_class=2)：车辆可能在行程中既有送货，也有取货，这会增加优化的难度
        glob_feats = super(MTVRPInitEmbeddingM, self)._global_feats(td)
        # 新的 backhaul_class 特征
        is_mixed_backhaul = (td["backhaul_class"] == 2).float()
        # 合并
        return torch.cat([glob_feats, is_mixed_backhaul[..., None]], -1)


class MTVRPInitEmbeddingFull(MTVRPInitEmbeddingRouteFinder):
    """
    This is the full embedding, including the Mixed Backhaul (MB) variants
    and Multi-depot (MD) variants as well.
    Node features:
        Customer features:
            - locs: x, y euclidean coordinates
            - demand_linehaul: demand of the nodes (delivery) (C)
            - demand_backhaul: demand of the nodes (pickup) (B)
            - time_windows: time window (TW)
            - service_time: service time of the nodes
        Depot features:
            - (end) time window of depots
            - x, y euclidean coordinates of depots (MD, optional)
    Global features:
        - open_route (O)
        - distance_limit (L)
        - backhaul_class (MB)
    This allows the network to learn the relationships between them.
    Note that the global features in practice are embedded in the depots, since
    they represent vehicle characteristics.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingRouteFinder, self).__init__(
            num_global_feats=4,  # x, y, open_route, distance_limit, time_window_depot, backhaul_class
            num_cust_feats=5,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        """Global features include the depot node(s) features"""
        num_depots = td["num_depots"].max().item()
        return torch.cat(
            [
                td["open_route"].float()[..., None].repeat(1, num_depots, 1),
                # td["locs"][:, :num_depots, :],
                td["distance_limit"][..., None].repeat(1, num_depots, 1),
                td["time_windows"][:, :num_depots, 1:2],
                (td["backhaul_class"] == 2).float()[..., None].repeat(1, num_depots, 1),
            ],
            -1,
        )

    def _cust_feats(self, td):
        num_depots = td["num_depots"].max().item()
        return torch.cat(
            (
                # td["locs"][..., num_depots:, :],
                td["demand_linehaul"][..., num_depots:, None],
                td["demand_backhaul"][..., num_depots:, None],
                td["time_windows"][..., num_depots:, :],
                td["service_time"][..., num_depots:, None],
            ),
            -1,
        )
