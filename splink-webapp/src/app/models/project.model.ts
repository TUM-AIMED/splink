import {BaseModel, IModelJson} from './base.model';

type ToolType = 'Select' | 'sPLINK'; // ADD THE TOOL NAME(S) HERE
type AlgorithmType = 'Select' | 'Logistic-Regression' | 'Linear-Regression' | 'Chi-Square'; // ADD THE ALGORITHM NAME(S) HERE

type StatusType = 'Created' | 'Parameters Ready' | 'Aggregating' | 'Done' | 'Aborted' | 'Failed';

export interface ProjectJson extends IModelJson {

  // Attributes common among all project types (tools)
  tool: ToolType;
  algorithm: AlgorithmType;
  name: string;
  description: string;
  status?: StatusType;
  step?: string;
  comm_round?: number;
  roles?: string[];
  token?: string;
  created_at?: string;

  // runtime stats related attributes (common among tools)
  client_computation?: number;
  client_network_send?: number;
  client_network_receive?: number;
  client_idle?: number;
  compensator_computation?: number;
  compensator_network_send?: number;
  server_computation?: number;
  runtime_total?: number;

  // traffic stats
  client_server?: string;
  server_client?: string;
  client_compensator?: string;
  compensator_server?: string;
  traffic_total?: string;

  // BEGIN sPLINK SPECIFIC ATTRIBUTES
  covariates: string;
  chunk_size?: number;
  max_iterations?: number;
  // BEGIN sPLINK SPECIFIC ATTRIBUTES

}

export class ProjectModel extends BaseModel<ProjectJson> {

  private _tool: ToolType;
  private _algorithm: AlgorithmType;
  private _name: string;
  private _description: string;
  private _status: StatusType;
  private _step: string;
  private _commRound: number;
  private _roles: string[];
  private _createdAt: Date;

  private _clientComputation: number;
  private _clientNetworkSend: number;
  private _clientNetworkReceive: number;
  private _clientIdle: number;
  private _compensatorComputation: number;
  private _compensatorNetworkSend: number;
  private _serverComputation: number;
  private _runtimeTotal: number;

  private _clientServer: string;
  private _serverClient: string;
  private _clientCompensator: string;
  private _compensatorServer: string;
  private _trafficTotal: string;

  // BEGIN sPLINK SPECIFIC ATTRIBUTES
  private _covariates: string
  private _chunkSize: number
  private _maxIterations: number
  // END sPLINK SPECIFIC ATTRIBUTES


  constructor() {
    super();
  }

  public async refresh(proj: ProjectJson) {
    this._id = proj.id;
    this._tool = proj.tool;
    this._algorithm = proj.algorithm;
    this._name = proj.name;
    this._description = proj.description;
    this._status = proj.status;
    this._step = proj.step;
    this._commRound = proj.comm_round;
    this._roles = proj.roles;
    this._createdAt = new Date(proj.created_at);

    this._clientComputation = proj.client_computation;
    this._clientNetworkSend = proj.client_network_send;
    this._clientNetworkReceive = proj.client_network_receive;
    this._clientIdle = proj.client_idle;
    this._compensatorComputation = proj.compensator_computation;
    this._compensatorNetworkSend = proj.compensator_network_send;
    this._serverComputation = proj.server_computation;
    this._runtimeTotal = proj.runtime_total;

    this._clientServer = proj.client_server;
    this._serverClient = proj.server_client;
    this._clientCompensator = proj.client_compensator;
    this._compensatorServer = proj.compensator_server;
    this._trafficTotal = proj.traffic_total;

    // BEGIN sPLINK SPECIFIC ATTRIBUTES
    this._covariates = proj.covariates
    this._chunkSize = proj.chunk_size
    this._maxIterations = proj.max_iterations
    // END sPLINK SPECIFIC ATTRIBUTES


  }

  public get tool(): ToolType {
    return this._tool;
  }

  public get algorithm(): AlgorithmType {
    return this._algorithm;
  }

  public get name(): string {
    return this._name;
  }

    public get description(): string {
    return this._description;
  }

  public get status(): StatusType {
    return this._status;
  }

  public get step(): string {
    return this._step;
  }

  public get commRound(): number {
    return this._commRound;
  }

  public get roles(): string[] {
    return this._roles;
  }

  public get createdAt(): Date {
    return this._createdAt;
  }

  public get clientComputation(): number {
    return this._clientComputation;
  }

  public get clientNetworkSend(): number {
    return this._clientNetworkSend;
  }

  public get clientNetworkReceive(): number {
    return this._clientNetworkReceive;
  }

  public get clientIdle(): number {
    return this._clientIdle;
  }

  public get compensatorComputation(): number {
    return this._compensatorComputation;
  }

  public get compensatorNetworkSend(): number {
    return this._compensatorNetworkSend;
  }

  public get serverComputation(): number {
    return this._serverComputation;
  }

  public get runtimeTotal(): number {
    return this._runtimeTotal;
  }

  public get clientServer(): string {
    return this._clientServer;
  }

  public get serverClient(): string {
    return this._serverClient;
  }

  public get clientCompensator(): string {
    return this._clientCompensator;
  }

  public get compensatorServer(): string {
    return this._compensatorServer;
  }

  public get trafficTotal(): string {
    return this._trafficTotal;
  }


  // BEGIN sPLINK SPECIFIC ATTRIBUTES
  public get covariates(): string{
    return this._covariates;
  }

  public get chunkSize(): number{
    return this._chunkSize;
  }

  public get maxIterations(): number{
    return this._maxIterations;
  }
  // END sPLINK SPECIFIC ATTRIBUTES


}
