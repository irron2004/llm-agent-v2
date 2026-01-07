import { TestQuestion } from "../types";

export const TEST_QUESTIONS: TestQuestion[] = [
  {
    id: "q001",
    question: "ECOLITE3000 설비에서 PM Chamber 내부 View Port 쪽에 Local Plasma 및 Arcing이 발생하는 원인은 무엇인가?",
    groundTruthDocIds: ["PLACEHOLDER_Q001"],
    category: "troubleshooting",
    difficulty: "hard",
  },
  {
    id: "q002",
    question: "GENEVA XP 설비에서 APC 관련 내용이 있는 GCB 번호를 알 수 있을까?",
    groundTruthDocIds: ["PLACEHOLDER_Q002"],
    category: "information",
    difficulty: "easy",
  },
  {
    id: "q003",
    question: "SEC SRD 라인의 EPA404 LL 관련해서 점검 이력을 정리해줄 수 있을까?",
    groundTruthDocIds: ["PLACEHOLDER_Q003"],
    category: "information",
    difficulty: "medium",
  },
  {
    id: "q004",
    question: "mySERVICE 이력 중 SEC SRD 라인의 EPA404호기 LL 점검 이력을 찾을 수 있을까?",
    groundTruthDocIds: ["PLACEHOLDER_Q004"],
    category: "information",
    difficulty: "medium",
  },
  {
    id: "q005",
    question: "SEC SRD 라인의 EPA404 LL 관련해서 MYSERVICE 점검 이력을 정리해줄 수 있을까?",
    groundTruthDocIds: ["PLACEHOLDER_Q005"],
    category: "information",
    difficulty: "medium",
  },
  {
    id: "q006",
    question: "SUPRA N Baffle 장착 시 Screw 체결 토크 스펙은 얼마인가?",
    groundTruthDocIds: ["PLACEHOLDER_Q006"],
    category: "setup",
    difficulty: "easy",
  },
  {
    id: "q007",
    question: "SUPRA N TM ROBOT ENDEFFECTOR 장착 시 Screw 체결 토크 스펙은 얼마인가?",
    groundTruthDocIds: ["PLACEHOLDER_Q007"],
    category: "setup",
    difficulty: "easy",
  },
  {
    id: "q008",
    question: "SUPRA III 설비에서 APC Pressure Hunting 발생 시 점검해야 할 포인트는 무엇인가?",
    groundTruthDocIds: ["PLACEHOLDER_Q008"],
    category: "troubleshooting",
    difficulty: "medium",
  },
  {
    id: "q009",
    question: "SUPRA N APC Position 이상 현상 발생 시 점검 포인트를 트러블슈팅 가이드 기반으로 알려줄 수 있을까?",
    groundTruthDocIds: ["PLACEHOLDER_Q009"],
    category: "troubleshooting",
    difficulty: "hard",
  },
  {
    id: "q010",
    question: "SUPRA Np에서 발생한 Issue 내용들을 GCB 기반으로 정리해줄 수 있을까?",
    groundTruthDocIds: ["PLACEHOLDER_Q010"],
    category: "information",
    difficulty: "medium",
  },
  {
    id: "q011",
    question: "SUPRA V에서 SUPRA Np로 개조된 설비의 설비 호기명은 무엇인가?",
    groundTruthDocIds: ["PLACEHOLDER_Q011"],
    category: "information",
    difficulty: "easy",
  },
  {
    id: "q012",
    question: "EPAGQ03에서 Source Unready Alarm이 발생한 이력에 대해 정리해줄 수 있을까?",
    groundTruthDocIds: ["PLACEHOLDER_Q012"],
    category: "information",
    difficulty: "medium",
  },
  {
    id: "q013",
    question: "INTEGER model에서 Main Rack Door Open Interlock 발생 시 해결 방법은 무엇인가?",
    groundTruthDocIds: ["PLACEHOLDER_Q013"],
    category: "troubleshooting",
    difficulty: "medium",
  },
  {
    id: "q014",
    question: "SUPRA Vplus 설비에서 APC Sensor의 Part Number는 무엇인가?",
    groundTruthDocIds: ["PLACEHOLDER_Q014"],
    category: "information",
    difficulty: "easy",
  },
  {
    id: "q015",
    question: "INTEGER Plus 설비에서 PM PIN Motor 교체 시 Pin 높이는 몇으로 설정해야 하며, S/W는 몇 번을 OFF 해야 하는가?",
    groundTruthDocIds: ["PLACEHOLDER_Q015"],
    category: "setup",
    difficulty: "hard",
  },
];

export const getQuestionsByCategory = (category: string): TestQuestion[] =>
  TEST_QUESTIONS.filter((q) => q.category === category);

export const getQuestionById = (id: string): TestQuestion | undefined =>
  TEST_QUESTIONS.find((q) => q.id === id);

export const CATEGORIES = [
  { value: "information", label: "정보조회" },
  { value: "troubleshooting", label: "문제해결" },
  { value: "setup", label: "설정/셋업" },
  { value: "maintenance", label: "유지보수" },
];
