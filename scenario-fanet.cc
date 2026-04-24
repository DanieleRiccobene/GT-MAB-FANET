/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include <ns3/lte-ue-net-device.h>
#include "ns3/mmwave-helper.h"
#include "ns3/epc-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/lte-helper.h"   
#include "ns3/energy-heuristic.h"
#include "ns3/propagation-loss-model.h"

// --- LIBRERIE ENERGIA (Aggiunte da scenario-fanet) ---
#include "ns3/energy-module.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/simple-device-energy-model.h"

using namespace ns3;
using namespace mmwave;

/**
 * Scenario Three - Ibrido FANET
 */

NS_LOG_COMPONENT_DEFINE ("ScenarioThreeFanetHybrid");



std::ofstream outFile;
void BsStateTrace(std::string filename, Ptr<LteEnbNetDevice> ltedev, Ptr<LteEnbRrc> lte_rrc, EnergySourceContainer sources,NetDeviceContainer mmWaveEnbDevs)
{
  static std::ofstream outFile;
    if (!outFile.is_open())
    {
    outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
    outFile << "Timestamp UNIX Id State EnergyFraction RemainingEnergy\n";
    }

    // Build cellId -> energy map
    std::map<uint16_t, std::pair<double,double>> energyByCell; // fraction, remainingJ
    for (uint32_t j = 0; j < sources.GetN(); ++j)
    {
    auto src = DynamicCast<BasicEnergySource>(sources.Get(j));
    auto dev = DynamicCast<MmWaveEnbNetDevice>(mmWaveEnbDevs.Get(j));
    uint16_t cellId = dev->GetCellId();
    energyByCell[cellId] = {src->GetEnergyFraction(), src->GetRemainingEnergy()};
    }

    auto entry = lte_rrc->GetAllowHandoverTo();
    for (auto it = entry.begin(); it != entry.end(); ++it)
    {
    uint16_t cellId = it->first;
    bool state = it->second;
    uint64_t ts = ltedev->GetStartTime() + Simulator::Now().GetMilliSeconds();

    double frac = 0.0, rem = 0.0;
    if (energyByCell.count(cellId))
    {
    frac = energyByCell[cellId].first;
    rem  = energyByCell[cellId].second;
    }

    outFile << Simulator::Now().GetSeconds() << " "
    << ts << " "
    << cellId << " "
    << state << " "
    << frac << " "
    << rem << "\n";
    }
}


void PrintGnuplottableUeListToFile (std::string filename) { /* Invariata */ }
void PrintGnuplottableEnbListToFile (std::string filename) { /* Invariata */ }
void PrintPosition (Ptr<Node> node) { /* Invariata */ }

double
EstimateCoverageRadiusMeters (Ptr<PropagationLossModel> pathlossModel,
                              Ptr<MmWaveEnbNetDevice> enbDev,
                              double rxThresholdDbm,
                              double maxDistanceMeters,
                              double distanceStepMeters,
                              double probeHeightMeters)
{
  NS_ASSERT_MSG (pathlossModel, "Pathloss model is null");
  NS_ASSERT_MSG (enbDev, "eNB device is null");

  Ptr<MobilityModel> enbMobility = enbDev->GetNode ()->GetObject<MobilityModel> ();
  NS_ASSERT_MSG (enbMobility, "eNB mobility model is null");

  Ptr<Node> probeNode = CreateObject<Node> ();
  Ptr<ConstantPositionMobilityModel> probeMobility = CreateObject<ConstantPositionMobilityModel> ();
  probeNode->AggregateObject (probeMobility);

  const Vector enbPos = enbMobility->GetPosition ();
  const double txPowerDbm = enbDev->GetPhy ()->GetTxPower ();

  double coverageRadius = 0.0;
  for (double d = distanceStepMeters; d <= maxDistanceMeters; d += distanceStepMeters)
    {
      probeMobility->SetPosition (Vector (enbPos.x + d, enbPos.y, probeHeightMeters));
      const double rxPowerDbm = pathlossModel->CalcRxPower (txPowerDbm, enbMobility, probeMobility);
      if (rxPowerDbm >= rxThresholdDbm)
        {
          coverageRadius = d;
        }
      else
        {
          break;
        }
    }

  return coverageRadius;
}

void
PrintEstimatedCoverageForMmWaveCells (Ptr<MmWaveHelper> mmwaveHelper,
                                      NetDeviceContainer mmWaveEnbDevs,
                                      double rxThresholdDbm)
{
  NS_ASSERT_MSG (mmwaveHelper, "MmWaveHelper is null");
  if (mmWaveEnbDevs.GetN () == 0)
    {
      NS_LOG_UNCOND ("No mmWave eNB devices found for coverage estimation.");
      return;
    }

  // Estimate along +X direction using the same altitude as the eNB node.
  const double maxDistanceMeters = 5000.0;
  const double distanceStepMeters = 5.0;
  Ptr<PropagationLossModel> pathlossModel = mmwaveHelper->GetPathLossModel (0);
  NS_ASSERT_MSG (pathlossModel, "Unable to retrieve pathloss model");

  NS_LOG_UNCOND ("--- Estimated mmWave cell coverage (RX threshold = "
                 << rxThresholdDbm << " dBm) ---");
  for (uint32_t i = 0; i < mmWaveEnbDevs.GetN (); ++i)
    {
      Ptr<MmWaveEnbNetDevice> enbDev = DynamicCast<MmWaveEnbNetDevice> (mmWaveEnbDevs.Get (i));
      if (!enbDev)
        {
          continue;
        }

      Ptr<MobilityModel> enbMobility = enbDev->GetNode ()->GetObject<MobilityModel> ();
      const double probeHeightMeters = enbMobility->GetPosition ().z;
      const double radiusMeters = EstimateCoverageRadiusMeters (pathlossModel,
                                                                enbDev,
                                                                rxThresholdDbm,
                                                                maxDistanceMeters,
                                                                distanceStepMeters,
                                                                probeHeightMeters);
      NS_LOG_UNCOND ("CellId " << enbDev->GetCellId () << " estimated radius: "
                               << radiusMeters << " m");
    }
}

// --- PARAMETRI GLOBALI (Invariati) ---
static ns3::GlobalValue g_bufferSize (
  "bufferSize",
  "RLC tx buffer size (MB)",
  ns3::UintegerValue (10),
  ns3::MakeUintegerChecker<uint32_t> ());

static ns3::GlobalValue g_rlcAmEnabled (
  "rlcAmEnabled",
  "If true, use RLC AM, else use RLC UM",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_enableTraces (
  "enableTraces",
  "If true, generate ns-3 traces",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_e2lteEnabled (
  "e2lteEnabled",
  "If true, send LTE E2 reports",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_e2nrEnabled (
  "e2nrEnabled",
  "If false, send NR E2 reports",
  ns3::BooleanValue (false),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_e2du (
  "e2du",
  "If true, send DU reports",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_e2cuUp (
  "e2cuUp",
  "If true, send CU-UP reports",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_e2cuCp (
  "e2cuCp",
  "If true, send CU-CP reports",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_trafficModel (
  "trafficModel",
  "Type of the traffic model",
  ns3::UintegerValue (0),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_   (
  "nBsNoUesAlloc",
  "Number of BS without UEs allocated",
  ns3::IntegerValue (-1),
  ns3::MakeIntegerChecker<int8_t> ());

static ns3::GlobalValue g_positionAllocator (
  "positionAllocator",
  "Type of the positionAllocator of UEs",
  ns3::UintegerValue (0),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_configuration (
  "configuration",
  "Set the wanted configuration",
  ns3::UintegerValue (1),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_dataRate (
  "dataRate",
  "Set the data rate",
  ns3::DoubleValue (0),
  ns3::MakeDoubleChecker<double> (0, 1));

static ns3::GlobalValue g_ues (
  "ues",
  "Number of UEs for each mmWave ENB.",
  ns3::UintegerValue (7),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_indicationPeriodicity (
  "indicationPeriodicity",
  "E2 Indication Periodicity",
  ns3::DoubleValue (0.1),
  ns3::MakeDoubleChecker<double> (0.01, 2.0));

static ns3::GlobalValue g_simTime (
  "simTime",
  "Simulation time in seconds",
  ns3::DoubleValue (1.9),
  ns3::MakeDoubleChecker<double> (0.1, 1000.0));

static ns3::GlobalValue g_reducedPmValues (
  "reducedPmValues",
  "If true, use a subset of the the pm containers",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_outageThreshold (
  "outageThreshold",
  "SNR threshold for outage events [dB]",
  ns3::DoubleValue (-1000.0),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_basicCellId (
  "basicCellId",
  "The next value will be the first cellId",
  ns3::UintegerValue (1),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_numberOfRaPreambles (
  "numberOfRaPreambles",
  "RACH preambles",
  ns3::UintegerValue (30),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_handoverMode (
  "handoverMode",
  "HO euristic",
  ns3::StringValue ("NoAuto"),
  ns3::MakeStringChecker ());

static ns3::GlobalValue g_e2TermIp (
  "e2TermIp",
  "The IP address of the RIC E2 termination",
  ns3::StringValue ("10.244.0.240"),
  ns3::MakeStringChecker ());

static ns3::GlobalValue g_enableE2FileLogging (
  "enableE2FileLogging",
  "Offline file logging",
  ns3::BooleanValue (true),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_controlFileName (
  "controlFileName",
  "The path to the control file",
  ns3::StringValue (""),
  ns3::MakeStringChecker ());

static ns3::GlobalValue q_useSemaphores (
  "useSemaphores",
  "Semaphores for external env control",
  ns3::BooleanValue (false),
  ns3::MakeBooleanChecker ());

static ns3::GlobalValue g_minSpeed (
  "minSpeed",
  "minimum UE speed in m/s",
  ns3::DoubleValue (2.0),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_maxSpeed (
  "maxSpeed",
  "maximum UE speed in m/s",
  ns3::DoubleValue (4.0),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_heuristic (
  "heuristicType",
  "Type of heuristic",
  ns3::IntegerValue (-1),
  ns3::MakeIntegerChecker<int8_t> ());

static ns3::GlobalValue g_probOn (
  "probOn",
  "Probability to turn BS ON",
  ns3::DoubleValue (0.6038),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_probIdle (
  "probIdle",
  "Probability to turn BS Idle",
  ns3::DoubleValue (0.3854),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_probSleep (
  "probSleep",
  "Probability to turn BS Sleep",
  ns3::DoubleValue (0.0107),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_probOff (
  "probOff",
  "Probability to turn BS Off",
  ns3::DoubleValue (0.0),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_sinrTh (
  "sinrTh",
  "SINR threshold",
  ns3::DoubleValue (73.0),
  ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue g_bsOn (
  "bsOn",
  "number of BS to turn ON",
  ns3::UintegerValue (2),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_bsIdle (
  "bsIdle",
  "number of BS to turn IDLE",
  ns3::UintegerValue (2),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_bsSleep (
  "bsSleep",
  "number of BS to turn Sleep",
  ns3::UintegerValue (2),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_bsOff (
  "bsOff",
  "number of BS to turn Off",
  ns3::UintegerValue (1),
  ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue g_nMmWaveEnbNodes (
  "nMmWaveEnbNodes",
  "Number of mmWave eNBs (drones)",
  ns3::UintegerValue (9), // default 9
  ns3::MakeUintegerChecker<uint8_t> ());


int main (int argc, char *argv[])
{
  LogComponentEnableAll (LOG_PREFIX_ALL);

  CommandLine cmd;
  cmd.Parse (argc, argv);

  bool harqEnabled = true;

  UintegerValue uintegerValue;
  IntegerValue integerValue;
  BooleanValue booleanValue;
  StringValue stringValue;
  DoubleValue doubleValue;

  GlobalValue::GetValueByName ("dataRate", doubleValue);
  double dataRateFromConf = doubleValue.Get ();
  GlobalValue::GetValueByName ("rlcAmEnabled", booleanValue);
  bool rlcAmEnabled = booleanValue.Get ();
  GlobalValue::GetValueByName ("bufferSize", uintegerValue);
  uint32_t bufferSize = uintegerValue.Get ();
  GlobalValue::GetValueByName ("basicCellId", uintegerValue);
  uint16_t basicCellId = uintegerValue.Get ();
  GlobalValue::GetValueByName ("enableTraces", booleanValue);
  bool enableTraces = booleanValue.Get ();
  GlobalValue::GetValueByName ("trafficModel", uintegerValue);
  uint8_t trafficModel = uintegerValue.Get ();
  //GlobalValue::GetValueByName ("nBsNoUesAlloc", integerValue);
  //int8_t nBsNoUesAlloc = integerValue.Get ();
  //GlobalValue::GetValueByName ("positionAllocator", uintegerValue);
  //uint8_t positionAllocator = uintegerValue.Get ();
  GlobalValue::GetValueByName ("outageThreshold",doubleValue);
  double outageThreshold = doubleValue.Get ();
  GlobalValue::GetValueByName ("handoverMode", stringValue);
  std::string handoverMode = stringValue.Get ();
  GlobalValue::GetValueByName ("e2TermIp", stringValue);
  std::string e2TermIp = stringValue.Get ();
  GlobalValue::GetValueByName ("enableE2FileLogging", booleanValue);
  bool enableE2FileLogging = booleanValue.Get ();
  //GlobalValue::GetValueByName ("minSpeed", doubleValue);
  //double minSpeed = doubleValue.Get ();
  //GlobalValue::GetValueByName ("maxSpeed", doubleValue);
  //double maxSpeed = doubleValue.Get ();
  GlobalValue::GetValueByName ("numberOfRaPreambles", uintegerValue);
  uint8_t numberOfRaPreambles = uintegerValue.Get ();

  // Heuristic parameters
  GlobalValue::GetValueByName ("heuristicType", integerValue);
  int8_t heuristicType = integerValue.Get ();
  GlobalValue::GetValueByName ("probOn", doubleValue);
  double probOn = doubleValue.Get ();
  GlobalValue::GetValueByName ("probIdle", doubleValue);
  double probIdle = doubleValue.Get ();
  GlobalValue::GetValueByName ("probSleep", doubleValue);
  double probSleep = doubleValue.Get ();
  GlobalValue::GetValueByName ("probOff", doubleValue);
  double probOff = doubleValue.Get ();
  GlobalValue::GetValueByName ("sinrTh", doubleValue);
  double sinrTh = doubleValue.Get ();
  GlobalValue::GetValueByName ("bsOn", uintegerValue);
  int bsOn = uintegerValue.Get ();
  GlobalValue::GetValueByName ("bsIdle", uintegerValue);
  int bsIdle = uintegerValue.Get ();
  GlobalValue::GetValueByName ("bsSleep", uintegerValue);
  int bsSleep = uintegerValue.Get ();
  GlobalValue::GetValueByName ("bsOff", uintegerValue);
  int bsOff = uintegerValue.Get ();

  GlobalValue::GetValueByName ("e2lteEnabled", booleanValue);
  bool e2lteEnabled = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2nrEnabled", booleanValue);
  bool e2nrEnabled = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2du", booleanValue);
  bool e2du = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2cuUp", booleanValue);
  bool e2cuUp = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2cuCp", booleanValue);
  bool e2cuCp = booleanValue.Get ();
  GlobalValue::GetValueByName ("reducedPmValues", booleanValue);
  bool reducedPmValues = booleanValue.Get ();
  GlobalValue::GetValueByName ("indicationPeriodicity", doubleValue);
  double indicationPeriodicity = doubleValue.Get ();
  GlobalValue::GetValueByName ("controlFileName", stringValue);
  std::string controlFilename = stringValue.Get ();
  GlobalValue::GetValueByName ("useSemaphores", booleanValue);
  bool useSemaphores = booleanValue.Get ();
  GlobalValue::GetValueByName ("nMmWaveEnbNodes", uintegerValue);
  uint8_t nMmWaveEnbNodes = uintegerValue.Get ();

  Config::SetDefault ("ns3::LteEnbNetDevice::UseSemaphores", BooleanValue (useSemaphores));
  Config::SetDefault ("ns3::LteEnbNetDevice::ControlFileName", StringValue(controlFilename));
  Config::SetDefault ("ns3::LteEnbNetDevice::E2Periodicity", DoubleValue (indicationPeriodicity));
  Config::SetDefault ("ns3::MmWaveEnbNetDevice::E2Periodicity", DoubleValue (indicationPeriodicity));
  Config::SetDefault ("ns3::MmWaveHelper::E2Periodicity", DoubleValue (indicationPeriodicity));
  Config::SetDefault ("ns3::MmWaveHelper::E2ModeLte", BooleanValue(e2lteEnabled));
  Config::SetDefault ("ns3::MmWaveHelper::E2ModeNr", BooleanValue(e2nrEnabled));
  Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableDuReport", BooleanValue(e2du));
  Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableCuUpReport", BooleanValue(e2cuUp));
  Config::SetDefault ("ns3::LteEnbNetDevice::EnableCuUpReport", BooleanValue(e2cuUp));
  Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableCuCpReport", BooleanValue(e2cuCp));
  Config::SetDefault ("ns3::LteEnbNetDevice::EnableCuCpReport", BooleanValue(e2cuCp));
  Config::SetDefault ("ns3::MmWaveEnbNetDevice::ReducedPmValues", BooleanValue (reducedPmValues));
  Config::SetDefault ("ns3::LteEnbNetDevice::ReducedPmValues", BooleanValue (reducedPmValues));
  Config::SetDefault ("ns3::LteEnbNetDevice::EnableE2FileLogging", BooleanValue (enableE2FileLogging));
  Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableE2FileLogging", BooleanValue (enableE2FileLogging));
  Config::SetDefault ("ns3::MmWaveEnbMac::NumberOfRaPreambles", UintegerValue (numberOfRaPreambles));
  Config::SetDefault ("ns3::MmWaveHelper::RlcAmEnabled", BooleanValue (rlcAmEnabled));
  Config::SetDefault ("ns3::MmWaveHelper::HarqEnabled", BooleanValue (harqEnabled));
  Config::SetDefault ("ns3::MmWaveHelper::UseIdealRrc", BooleanValue (true));
  Config::SetDefault ("ns3::MmWaveHelper::BasicCellId", UintegerValue (basicCellId));
  Config::SetDefault ("ns3::MmWaveHelper::BasicImsi", UintegerValue ((basicCellId-1)));
  Config::SetDefault ("ns3::MmWaveHelper::E2TermIp", StringValue (e2TermIp));
  Config::SetDefault ("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue (harqEnabled));
  Config::SetDefault ("ns3::MmWavePhyMacCommon::NumHarqProcess", UintegerValue (100));
  Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue (MilliSeconds (100.0)));
  Config::SetDefault ("ns3::ThreeGppChannelConditionModel::UpdatePeriod", TimeValue (MilliSeconds (100)));
  Config::SetDefault ("ns3::LteRlcAm::ReportBufferStatusTimer", TimeValue (MilliSeconds (10.0)));
  Config::SetDefault ("ns3::LteRlcUmLowLat::ReportBufferStatusTimer", TimeValue (MilliSeconds (10.0)));
  Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (bufferSize * 1024 * 1024));
  Config::SetDefault ("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue (bufferSize * 1024 * 1024));
  Config::SetDefault ("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue (bufferSize * 1024 * 1024));
  Config::SetDefault ("ns3::LteEnbRrc::OutageThreshold", DoubleValue (outageThreshold));
  Config::SetDefault ("ns3::LteEnbRrc::SecondaryCellHandoverMode", StringValue (handoverMode));

  double bandwidth;
  double centerFrequency;
  int numAntennasMcUe;
  int numAntennasMmWave;
  std::string dataRate;

  GlobalValue::GetValueByName ("configuration", uintegerValue);
  uint8_t configuration = uintegerValue.Get ();
  switch (configuration)
    {
    case 0:
      centerFrequency = 850e6; bandwidth = 20e6; numAntennasMcUe = 1; numAntennasMmWave = 1; dataRate = (dataRateFromConf == 0 ? "1.5Mbps" : "4.5Mbps"); break;
    case 1:
      centerFrequency = 3.5e9; bandwidth = 20e6; numAntennasMcUe = 1; numAntennasMmWave = 1; dataRate = (dataRateFromConf == 0 ? "1.5Mbps" : "4.5Mbps"); break;
    case 2:
      centerFrequency = 28e9; bandwidth = 100e6; numAntennasMcUe = 16; numAntennasMmWave = 64; dataRate = (dataRateFromConf == 0 ? "15Mbps" : "45Mbps"); break;
    default: NS_FATAL_ERROR ("Configuration not recognized" << configuration); break;
    }

  Config::SetDefault ("ns3::MmWavePhyMacCommon::Bandwidth", DoubleValue (bandwidth));
  Config::SetDefault ("ns3::MmWavePhyMacCommon::CenterFreq", DoubleValue (centerFrequency));

  Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper> ();
  mmwaveHelper->SetPathlossModelType ("ns3::ThreeGppUmaPropagationLossModel");
  mmwaveHelper->SetChannelConditionModelType ("ns3::ThreeGppUmaChannelConditionModel");

  mmwaveHelper->SetUePhasedArrayModelAttribute("NumColumns", UintegerValue(std::sqrt(numAntennasMcUe)));
  mmwaveHelper->SetUePhasedArrayModelAttribute("NumRows", UintegerValue(std::sqrt(numAntennasMcUe)));
  mmwaveHelper->SetEnbPhasedArrayModelAttribute("NumColumns",UintegerValue(std::sqrt(numAntennasMmWave)));
  mmwaveHelper->SetEnbPhasedArrayModelAttribute("NumRows", UintegerValue(std::sqrt(numAntennasMmWave)));

  Ptr<MmWavePointToPointEpcHelper> epcHelper = CreateObject<MmWavePointToPointEpcHelper> ();
  mmwaveHelper->SetEpcHelper (epcHelper);

  //  uint8_t nMmWaveEnbNodes = 3;
  uint8_t nLteEnbNodes = 1;
  GlobalValue::GetValueByName ("ues", uintegerValue);
  uint32_t ues = uintegerValue.Get ();
  uint8_t nUeNodes = ues;

  Ptr<Node> pgw = epcHelper->GetPgwNode ();
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (2500));
  p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.010)));
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

  NodeContainer ueNodes;
  NodeContainer mmWaveEnbNodes;
  NodeContainer lteEnbNodes;
  NodeContainer allEnbNodes;
  mmWaveEnbNodes.Create (nMmWaveEnbNodes);
  lteEnbNodes.Create (nLteEnbNodes);
  ueNodes.Create (nUeNodes);
  allEnbNodes.Add (lteEnbNodes);
  allEnbNodes.Add (mmWaveEnbNodes);

    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator> ();
  
  switch(nMmWaveEnbNodes)
  {
    case 3: {
              // Middle Row (3 drones)
              enbPositionAlloc->Add (Vector (0.0,    2000.0, 25.0)); // Mid Left
              enbPositionAlloc->Add (Vector (2000.0, 2000.0, 25.0)); // Center
              enbPositionAlloc->Add (Vector (4000.0, 2000.0, 25.0)); // Mid Right
    }
    break;

    case 5: {
              enbPositionAlloc->Add (Vector (0.0,    4000.0, 25.0)); // Top Left
              enbPositionAlloc->Add (Vector (4000.0, 4000.0, 25.0)); // Top Right

              enbPositionAlloc->Add (Vector (2000.0, 2000.0, 25.0)); // Center

              enbPositionAlloc->Add (Vector (0.0,    0.0,    25.0)); // Bottom Left
              enbPositionAlloc->Add (Vector (4000.0, 0.0,    25.0)); // Bottom Right
    }
    break;

    case 7: {
              enbPositionAlloc->Add (Vector (0.0,    4000.0, 25.0)); // Top Left
              enbPositionAlloc->Add (Vector (4000.0, 4000.0, 25.0)); // Top Right

              // Middle Row (3 drones)
              enbPositionAlloc->Add (Vector (0.0,    2000.0, 25.0)); // Mid Left
              enbPositionAlloc->Add (Vector (2000.0, 2000.0, 25.0)); // Center
              enbPositionAlloc->Add (Vector (4000.0, 2000.0, 25.0)); // Mid Right

              // Bottom Row (2 drones)
              enbPositionAlloc->Add (Vector (0.0,    0.0,    25.0)); // Bottom Left
              enbPositionAlloc->Add (Vector (4000.0, 0.0,    25.0)); // Bottom Right
    }
    break;

    case 9: {
               // 3x3 grid: X,Y ∈ {0, 2000, 4000}, Z = 3 m
              for (double x_pos = 0.0; x_pos <= 4000.0; x_pos += 2000.0)
              {
                for (double y_pos = 0.0; y_pos <= 4000.0; y_pos += 2000.0)
                  {
                    enbPositionAlloc->Add(Vector (x_pos, y_pos, 25.0));
                  }
              }
    }
    break;
  }
 

  MobilityHelper enbmobility;
  enbmobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel"); // without specifying any speed the drones will be stationary
  enbmobility.SetPositionAllocator (enbPositionAlloc);
  enbmobility.Install (allEnbNodes);

  // =========================================================================
  // --- POSIZIONAMENTO UE (GEOMETRIA FANET) ---
  // =========================================================================
  MobilityHelper uemobility;
  uemobility.SetPositionAllocator ("ns3::RandomRectanglePositionAllocator",
                                   "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=4000.0]"),
                                   "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=4000.0]"));
  uemobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  uemobility.Install (ueNodes);

  NetDeviceContainer lteEnbDevs = mmwaveHelper->InstallLteEnbDevice (lteEnbNodes);
  NetDeviceContainer mmWaveEnbDevs = mmwaveHelper->InstallEnbDevice (mmWaveEnbNodes);
  NetDeviceContainer mcUeDevs = mmwaveHelper->InstallMcUeDevice (ueNodes);

  const double rxCoverageThresholdDbm = -100.0;
  PrintEstimatedCoverageForMmWaveCells (mmwaveHelper, mmWaveEnbDevs, rxCoverageThresholdDbm);

  internet.Install (ueNodes);
  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (mcUeDevs));
  
  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      Ptr<Node> ueNode = ueNodes.Get (u);
      Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  mmwaveHelper->AddX2Interface (lteEnbNodes, mmWaveEnbNodes);
  // mmwaveHelper->AttachToClosestEnb (mcUeDevs, mmWaveEnbDevs, lteEnbDevs);
  // =========================================================================
  // --- CUSTOM ATTACHMENT WITH CAPACITY LIMIT ---
  // =========================================================================
  uint32_t maxUesPerBs = 10; // Enforce maximum number of UEs per drone
  std::map<uint32_t, uint32_t> uesPerBs; 

  for (uint32_t u = 0; u < mcUeDevs.GetN (); ++u)
    {
      Ptr<NetDevice> ueDev = mcUeDevs.Get (u);
      Ptr<MobilityModel> ueMob = ueDev->GetNode ()->GetObject<MobilityModel> ();
      
      Ptr<NetDevice> bestBs = nullptr;
      uint32_t bestBsIndex = 0;
      double minDistance = std::numeric_limits<double>::max ();
      
      for (uint32_t b = 0; b < mmWaveEnbDevs.GetN (); ++b)
        {
          // Skip this base station if it has reached the max connection limit
          if (uesPerBs[b] >= maxUesPerBs) continue;
          
          Ptr<NetDevice> enbDev = mmWaveEnbDevs.Get (b);
          Ptr<MobilityModel> enbMob = enbDev->GetNode ()->GetObject<MobilityModel> ();
          double distance = ueMob->GetDistanceFrom (enbMob);
          
          if (distance < minDistance)
            {
              minDistance = distance;
              bestBs = enbDev;
              bestBsIndex = b;
            }
        }
        
      if (bestBs)
        {
          // Attach the UE to the selected mmWave eNB and the primary LTE eNB
          mmwaveHelper->Attach (ueDev, bestBs, lteEnbDevs.Get (0));
          uesPerBs[bestBsIndex]++;
        }
      else
        {
          NS_LOG_UNCOND ("UE " << u << " could not attach: All nearby BSs reached the maximum connection limit of " << maxUesPerBs << ".");
        }
    }

  // =========================================================================
  // --- ENERGY MODEL (FANET BATTERY) ---
  // =========================================================================
  BasicEnergySourceHelper energySourceHelper;
  energySourceHelper.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (10000.0)); // declaring battery capacity
  // Installed on the Drones
  EnergySourceContainer sources = energySourceHelper.Install (mmWaveEnbNodes); // defining the source in drones
  
  // =========================================================================
  DeviceEnergyModelContainer deviceModels;
  for (uint32_t i = 0; i < mmWaveEnbDevs.GetN (); ++i) // defining the device model in drones
    {
      Ptr<SimpleDeviceEnergyModel> model = CreateObject<SimpleDeviceEnergyModel> ();
      model->SetEnergySource (sources.Get (i));
      model->SetNode (mmWaveEnbNodes.Get (i));
      model->SetCurrentA (0.01);
      // Connect model and source
      sources.Get (i)->AppendDeviceEnergyModel (model);
      deviceModels.Add (model);
    }



  // --- TRAFFICO (Invariato da Scenario Three Grid) ---
  uint16_t portTcp = 50000;
  Address sinkLocalAddressTcp (InetSocketAddress (Ipv4Address::GetAny (), portTcp));
  PacketSinkHelper sinkHelperTcp ("ns3::TcpSocketFactory", sinkLocalAddressTcp);
  AddressValue serverAddressTcp (InetSocketAddress (remoteHostAddr, portTcp));

  uint16_t portUdp = 60000;
  Address sinkLocalAddressUdp (InetSocketAddress (Ipv4Address::GetAny (), portUdp));
  PacketSinkHelper sinkHelperUdp ("ns3::UdpSocketFactory", sinkLocalAddressUdp);
  AddressValue serverAddressUdp (InetSocketAddress (remoteHostAddr, portUdp));

  ApplicationContainer sinkApp;
  sinkApp.Add (sinkHelperTcp.Install (remoteHost));
  sinkApp.Add (sinkHelperUdp.Install (remoteHost));

  OnOffHelper clientHelperTcp ("ns3::TcpSocketFactory", Address ());
  clientHelperTcp.SetAttribute ("Remote", serverAddressTcp);
  clientHelperTcp.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperTcp.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperTcp.SetAttribute ("DataRate", StringValue (dataRate));
  clientHelperTcp.SetAttribute ("PacketSize", UintegerValue (1280));

  OnOffHelper clientHelperTcp150 ("ns3::TcpSocketFactory", Address ());
  clientHelperTcp150.SetAttribute ("Remote", serverAddressTcp);
  clientHelperTcp150.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperTcp150.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperTcp150.SetAttribute ("DataRate", StringValue ("150kbps"));
  clientHelperTcp150.SetAttribute ("PacketSize", UintegerValue (1280));

  OnOffHelper clientHelperTcp750 ("ns3::TcpSocketFactory", Address ());
  clientHelperTcp750.SetAttribute ("Remote", serverAddressTcp);
  clientHelperTcp750.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperTcp750.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperTcp750.SetAttribute ("DataRate", StringValue ("750kbps"));
  clientHelperTcp750.SetAttribute ("PacketSize", UintegerValue (1280));

  OnOffHelper clientHelperUdp ("ns3::UdpSocketFactory", Address ());
  clientHelperUdp.SetAttribute ("Remote", serverAddressUdp);
  clientHelperUdp.SetAttribute ("OnTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperUdp.SetAttribute ("OffTime", StringValue ("ns3::ExponentialRandomVariable"));
  clientHelperUdp.SetAttribute ("DataRate", StringValue (dataRate));
  clientHelperUdp.SetAttribute ("PacketSize", UintegerValue (1280));

  ApplicationContainer clientApp;
  switch (trafficModel)
    {
      case 0: {
        for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
          {
            PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
            sinkApp.Add (dlPacketSinkHelper.Install (ueNodes.Get (u)));
            UdpClientHelper dlClient (ueIpIface.GetAddress (u), 1234);
            dlClient.SetAttribute ("Interval", TimeValue (MicroSeconds (500)));
            dlClient.SetAttribute ("MaxPackets", UintegerValue (UINT32_MAX));
            dlClient.SetAttribute ("PacketSize", UintegerValue (1280));
            clientApp.Add (dlClient.Install (remoteHost));
          }
      }
      break;

      case 1: {
        for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
          {
            if (u % 2 == 0)
              {
                if (u % 4 == 0) clientApp.Add (clientHelperTcp.Install (ueNodes.Get (u)));
                else clientApp.Add (clientHelperUdp.Install (ueNodes.Get (u)));
              }
            else
              {
                PacketSinkHelper dlPacketSinkHelper ( "ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
                sinkApp.Add (dlPacketSinkHelper.Install (ueNodes.Get (u)));
                UdpClientHelper dlClient (ueIpIface.GetAddress (u), 1234);
                dlClient.SetAttribute ("Interval", TimeValue (MicroSeconds (500)));
                dlClient.SetAttribute ("MaxPackets", UintegerValue (UINT32_MAX));
                dlClient.SetAttribute ("PacketSize", UintegerValue (1280));
                clientApp.Add (dlClient.Install (remoteHost));
              }
          }
      }
      break;

      case 2: {
        for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
          {
            if (u % 2 == 0) clientApp.Add (clientHelperTcp.Install (ueNodes.Get (u)));
            else clientApp.Add (clientHelperUdp.Install (ueNodes.Get (u)));
          }
      }
      break;

      case 3: { 
        for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
          {
            if (u % 4 == 0)
              {
                PacketSinkHelper dlPacketSinkHelper ( "ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 1234));
                sinkApp.Add (dlPacketSinkHelper.Install (ueNodes.Get (u)));
                UdpClientHelper dlClient (ueIpIface.GetAddress (u), 1234);
                dlClient.SetAttribute ("MaxPackets", UintegerValue (UINT32_MAX));
                dlClient.SetAttribute ("PacketSize", UintegerValue (1280));
                if (configuration == 2) dlClient.SetAttribute ("Interval", TimeValue (MicroSeconds (250)));
                else dlClient.SetAttribute ("Interval", TimeValue (MicroSeconds (500)));
                clientApp.Add (dlClient.Install (remoteHost));
              }
            else if (u % 4 == 1)
              {
                if (configuration == 2) clientHelperTcp.SetAttribute ("DataRate", StringValue ("20Mbps"));
                clientApp.Add (clientHelperTcp.Install (ueNodes.Get (u)));
              }
            else if (u % 4 == 2) clientApp.Add (clientHelperTcp750.Install (ueNodes.Get (u)));
            else if (u % 4 == 3) clientApp.Add (clientHelperTcp150.Install (ueNodes.Get (u)));
          }
        break;
      }
      default: NS_FATAL_ERROR ("Traffic model not recognized");
    }

  GlobalValue::GetValueByName ("simTime", doubleValue);
  double simTime = doubleValue.Get ();
  sinkApp.Start (Seconds (0));
  clientApp.Start (MilliSeconds (100));
  clientApp.Stop (Seconds (simTime - 0.1));

  int BsStatus[4] = {bsOn, bsIdle, bsSleep, bsOff};
  if (bsIdle == 0) { BsStatus[1] = bsOn; BsStatus[0] = 0; }

  Ptr<EnergyHeuristic> energyHeur=CreateObject<EnergyHeuristic>();
  

  switch (heuristicType)
    {
      case -1: {
        NS_LOG_UNCOND ("Running the scenario with External CMAB Control");
      }
      break;

      case 0: {
        for (double i = 0.0; i < simTime; i = i + indicationPeriodicity)
          {
            for (int j = 0; j < nMmWaveEnbNodes; j++)
              {
                Ptr<MmWaveEnbNetDevice> mmdev = DynamicCast<MmWaveEnbNetDevice> (mmWaveEnbDevs.Get (j));
                Ptr<LteEnbNetDevice> ltedev = DynamicCast<LteEnbNetDevice> (lteEnbDevs.Get (0));
                Simulator::Schedule (Seconds (i), &EnergyHeuristic::ProbabilityState, energyHeur, probOn, probIdle, probSleep, probOff, mmdev, ltedev);
              }
          }
      }
      break;

      case 1: {
        for (double i = 0.0; i < simTime; i = i + indicationPeriodicity)
          {
            for (int j = 0; j < nMmWaveEnbNodes && bsOn!=0; j++)
              {
                Ptr<MmWaveEnbNetDevice> mmdev = DynamicCast<MmWaveEnbNetDevice> (mmWaveEnbDevs.Get (j));
                Simulator::Schedule (Seconds (i), &EnergyHeuristic::CountBestUesSinr, energyHeur, sinrTh, mmdev);
              }
            Ptr<LteEnbNetDevice> ltedev = DynamicCast<LteEnbNetDevice> (lteEnbDevs.Get (0));
            Simulator::Schedule (Seconds (i), &EnergyHeuristic::TurnOnBsSinrPos, energyHeur, nMmWaveEnbNodes, mmWaveEnbDevs, "static", BsStatus, ltedev);
          }
      }
      break;

      case 2: {
        for (double i = 0.0; i < simTime; i = i + indicationPeriodicity)
          {
            for (int j = 0; j < nMmWaveEnbNodes && bsOn!=0; j++)
              {
                Ptr<MmWaveEnbNetDevice> mmdev = DynamicCast<MmWaveEnbNetDevice> (mmWaveEnbDevs.Get (j));
                Simulator::Schedule (Seconds (i), &EnergyHeuristic::CountBestUesSinr, energyHeur, sinrTh, mmdev);
              }
            Ptr<LteEnbNetDevice> ltedev = DynamicCast<LteEnbNetDevice> (lteEnbDevs.Get (0));
            Simulator::Schedule (Seconds (i), &EnergyHeuristic::TurnOnBsSinrPos, energyHeur, nMmWaveEnbNodes, mmWaveEnbDevs, "dynamic", BsStatus, ltedev);
          }
      }
      break;
      
      default: {
        NS_FATAL_ERROR ("Heuristic type not recognized");
      }
      break;
    }

  if (enableTraces)
  {
    mmwaveHelper->EnableTraces ();
  }  

  // --- TRACCIAMENTO KPI (Invariato per evitare il KeyError in Python) ---
  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  lteHelper->Initialize ();
  lteHelper->EnablePhyTraces ();
  lteHelper->EnableMacTraces ();

  // Esporta le posizioni (I droni si muoveranno, quindi questi file conterranno solo lo spawn)
  PrintGnuplottableUeListToFile ("ues.txt");
  PrintGnuplottableEnbListToFile ("enbs.txt");
  
  Ptr<LteEnbNetDevice> ltedev = DynamicCast<LteEnbNetDevice> (lteEnbDevs.Get (0));
  Ptr<LteEnbRrc> lte_rrc = ltedev->GetRrc ();  
  

  for (double i = 0.0; i < simTime; i += indicationPeriodicity)
{
  Simulator::Schedule(Seconds(i), BsStateTrace, "bsState.txt",
                      ltedev, lte_rrc, sources, mmWaveEnbDevs);
}

  bool run = true;
  if (run)
    {
      NS_LOG_UNCOND ("Simulation time is " << simTime << " seconds ");
      Simulator::Stop (Seconds (simTime));
      NS_LOG_INFO ("Run Simulation.");
      Simulator::Run ();
    }

  NS_LOG_INFO (lteHelper);

  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
  return 0;
}